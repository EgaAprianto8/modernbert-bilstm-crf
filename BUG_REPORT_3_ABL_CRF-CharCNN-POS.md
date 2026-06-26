# Code Review: 3_ABL_CRF-CharCNN-POS.ipynb

## Executive Summary
Notebook ini memiliki **beberapa bug kritis** yang explains CUDA OOM dan error lainnya saat dijalankan di vast.ai. Bug utama mencakup undefined config variables, memory inefficiency, dan logic errors.

---

## BUGS DITEMUKAN

### 🔴 CRITICAL BUGS

#### BUG 1: Undefined Config Variables
**Location:** Line 624
```python
self.self_attn = nn.MultiheadAttention(
    embed_dim=classifier_in, num_heads=cfg.NUM_HEADS, dropout=cfg.ATTN_DROPOUT, batch_first=True,
)
```
**Problem:** `cfg.NUM_HEADS` dan `cfg.ATTN_DROPOUT` **TIDAK PERNAH didefinisikan** dalam class CFG (line 47-121).

**Impact:** 
- Jika `USE_SELF_ATTENTION=True` → AttributeError saat inisialisasi
- Notebook ini punya `USE_SELF_ATTENTION=False` jadi tidak crash sekarang, tapi fitur tidak bisa diaktifkan

**Severity:** CRITICAL (menghambat fitur)

**Fix:**
```python
# Tambahkan ke CFG class:
NUM_HEADS: int = 8
ATTN_DROPOUT: float = 0.1
```

---

#### BUG 2: CUDA OOM - Batch Size + Sequence Length
**Location:** Line 55-56
```python
MAX_LENGTH: int = 8192
BATCH_SIZE: int = 8
```
**Problem:** ModernBERT-Large dengan seq_len=8192 dan batch_size=8 adalah kombinasi yang sangat berat di GPU.

**Memory calculation:**
- ModernBERT-Large: ~435M params
- Hidden size: 1024
- Sequence length: 8192 tokens
- Per sample memory (BF16): ~1024 × 8192 × 2 bytes = ~16MB per sample
- Attention matrix: 8192² × 2 / 1024 = ~131MB per sample
- With batch size 8: bisa >1GB hanya untuk activations

**Impact:** CUDA Out of Memory, terutama di GPU dengan VRAM < 24GB

**Fix Options:**
```python
# Option 1: Kurangi batch size
BATCH_SIZE: int = 2  # Atau bahkan 1

# Option 2: Kurangi MAX_LENGTH
MAX_LENGTH: int = 2048  # atau 4096

# Option 3: Gradient checkpointing (sudah ada tapi perlu verify)
GRADIENT_CHECKPOINTING: bool = True

# Option 4: Kurangi encoder layers untuk ablations
ENCODER_LAYERS_TO_USE: int = 12  # dari 24 layer ModernBERT-Large
```

---

#### BUG 3: Flash Attention Compatibility Issue
**Location:** Line 100, 574
```python
USE_FLASH_ATTN: bool = True
```
**Problem:** Flash Attention 2 tidak compatible dengan semua GPU/CUDA versions. Jika GPU lama atau CUDA version tidak support, model loading akan gagal.

**Impact:** Model loading error atau silent fallback ke eager attention yang consumption lebih banyak memory

**Fix:**
```python
# Cek kompatibilitas sebelum menggunakan flash attention
if cfg.USE_FLASH_ATTN and torch.cuda.is_available():
    # Test flash attention compatibility
    try:
        test_result = torch.nn.functional.scaled_dot_product_attention(
            torch.zeros(2, 2, 2, 2, device='cuda', dtype=torch.float16),
            torch.zeros(2, 2, 2, 2, device='cuda', dtype=torch.float16),
            torch.zeros(2, 2, 2, 2, device='cuda', dtype=torch.float16),
            is_causal=True
        )
        USE_FLASH = True
    except:
        USE_FLASH = False
        print("⚠ Flash attention not available, using eager attention")
```

---

### 🟠 HIGH SEVERITY BUGS

#### BUG 4: CharCNN Padding dengan MAX_WORD_LEN=20
**Location:** Line 527
```python
self.convs = nn.ModuleList([
    nn.Conv1d(embed_dim, f, kernel_size=k, padding='same')
    for k, f in zip(kernel_sizes, filter_sizes)
])
```
**Problem:** `padding='same'` di Conv1d membutuhkan input length ≥ kernel_size. Dengan `MAX_WORD_LEN=20` dan kernel sizes (4, 3, 3), ini OK. Tapi sebenarnya ada edge case.

**Impact:** Jika ada kata dengan panjang > MAX_WORD_LEN (20), padding dilakukan tapi tidak ter-handle dengan benar di words_to_char_ids

**Fix:** Pastikan semua kata dipotong ke MAX_WORD_LEN sebelum embedding (sudah dilakukan di line 403 tapi perlu verify)

---

#### BUG 5: CRF Labels Clamping
**Location:** Line 764-766
```python
safe = labels.clone()
safe[safe == -100] = 0
safe = torch.clamp(safe, min=0, max=self.crf.num_tags - 1)
```
**Problem:** CRF torchcrf library tidak handle `-100` labels dengan benar. Label `-100` di-clamp ke `0`, tapi ini mungkin bukan behavior yang diinginkan.

**Impact:** Token yang di-mask (padding) bisa contribute negative loss yang tidak seharusnya

**Fix:**
```python
# Buat mask untuk valid tokens saja
valid_mask = labels != -100
# CRF hanya compute loss untuk valid tokens
# Atau gunakan custom CRF wrapper yang handle -100
```

---

#### BUG 6: Ambiguous Early Stopping Logic
**Location:** Line 1022-1034
```python
# Early stopping: berdasarkan DEV LOSS (lebih sensitif dari F1)
if dev_loss is not None:
    if dev_loss > best_dev_loss - 0.01:  # tidak turun signifikan
        patience_counter += 1
    else:
        best_dev_loss = dev_loss
        patience_counter = 0
else:
    # Fallback: stop jika dev F1 tidak improvement
    if not is_best:
        patience_counter += 1
    else:
        patience_counter = 0
```
**Problem:** Logic early stopping memiliki threshold hardcoded `0.01` yang tidak bisa dikonfigurasi

**Impact:** Training bisa berhenti prematur atau terlalu lama tergantung data

**Fix:**
```python
# Tambahkan konfigurasi
LOSS_THRESHOLD: float = 0.01  # di CFG

# Gunakan konfigurasi
if dev_loss > best_dev_loss - cfg.LOSS_THRESHOLD:
    patience_counter += 1
```

---

### 🟡 MEDIUM SEVERITY BUGS

#### BUG 7: Subword Strategy -100 in CRF
**Location:** Line 419-426
```python
def subword_label(tag, is_first):
    if is_first:
        return label2id.get(tag, 0)
    if cfg.SUBWORD_STRATEGY == 'all_subwords':
        if tag.startswith('B-'):
            tag = 'I-' + tag[2:]
        return label2id.get(tag, 0)
    return -100
```
**Problem:** Saat `SUBWORD_STRATEGY != 'all_subwords'`, function return `-100`. Tapi CRF tidak bisa handle `-100` labels dengan benar.

**Impact:** Jika switch strategy, CRF akan crash atau give incorrect predictions

**Fix:** Pastikan CRF wrapper handle `-100` dengan benar

---

#### BUG 8: Collate Function - Inconsistent Padding
**Location:** Line 496
```python
out['case_ids'] = torch.stack([F.pad(x['case_ids'], (0, ml - len(x['case_ids'])), value=case2id['PADDING_TOKEN']) for x in batch])
```
**Problem:** case_ids di-pad dengan `case2id['PADDING_TOKEN']=7`, tapi di dataset creation (line 466-467), case_ids di-set dari `case_word[wid]` yang tidak termasuk padding token.

**Impact:** Padding tokens untuk casing tidak konsisten

---

#### BUG 9: GPU Device Assignment Duplicated
**Location:** Line 133-134, 918
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cell 3
# ...
device = get_device()  # cell 8, inside train_one_seed
```
**Problem:** Device di-assign dua kali di tempat berbeda. Jika dipanggil dari context berbeda, bisa inconsistency.

**Fix:** Gunakan single source of truth
```python
DEVICE = get_device()  # global constant
```

---

#### BUG 10: NERDataset Re-creation in Training
**Location:** Line 936
```python
train_ds_local = NERDataset(sent_train, lab_train, tokenizer, cfg.MAX_LENGTH)
```
**Problem:** Dataset di-recreate setiap kali `train_one_seed` dipanggil. Ini sudah dilakukan di cell 5-6. Double work.

**Impact:** Wasteful computation untuk POS tagging dan preprocessing

**Fix:** Reuse dataset yang sudah dibuat

---

### 🟢 LOW SEVERITY / WARNINGS

#### WARNING 1: Deprecated nltk.download
**Location:** Line 34-38
**Problem:** `nltk.download()` di-loop untuk setiap package bisa lambat dan deprecated

**Fix:**
```python
import nltk
nltk.data.find('taggers/averaged_perceptron_tagger_eng')
```

---

#### WARNING 2: StableAdamW Import
**Location:** Line 892-894
**Problem:** `pytorch_optimizer` tidak standard library, harus di-install

**Note:** Ini sudah di-handle dengan ImportError yang jelas

---

## RECOMMENDATIONS

### Priority 1: Fix OOM Issues
```python
# Sesuaikan dengan GPU memory Anda
BATCH_SIZE: int = 1  # atau 2
MAX_LENGTH: int = 2048  # atau 4096
```

### Priority 2: Add Missing Config
```python
# Di CFG class:
NUM_HEADS: int = 8
ATTN_DROPOUT: float = 0.1
```

### Priority 3: Add Memory Monitoring
```python
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.max_memory_allocated()/1e9:.2f}GB max")
```

### Priority 4: Verify CRF -100 Handling
Jika Anda mengalami issues dengan CRF loss, pertimbangkan untuk menggunakan CE loss sebagai fallback

---

## TESTING RECOMMENDATIONS

1. **Test dengan batch_size=1, max_length=512 dulu** untuk verify model bisa run
2. **Gradually increase** batch_size dan max_length
3. **Monitor GPU memory** dengan `nvidia-smi` selama training
4. **Test flash attention fallback** jika OOM masih terjadi