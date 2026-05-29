# Vietnamese Dishes Recognition AI

Du an nay dung mo hinh CNN bang TensorFlow/Keras de nhan dien mon an Viet Nam tu webcam hoac anh import. Ung dung chinh nam trong `gui.py` va model da train duoc luu o `model.keras`.

## Mon an model dang ho tro

Theo cau hinh trong `gui.py`, model hien tai du doan 5 lop:

- `BÁNH CUỐN`
- `BÁNH MÌ`
- `BÁNH XÈO`
- `BÚN ĐẬU MẮM TÔM`
- `BÁNH CANH`

Luu y: danh sach lop ma GUI dung duoc khai bao truc tiep trong `gui.py`.

## Cau truc thu muc

```text
VIETNAMESE DISHES/
├── gui.py          # Ung dung giao dien nhan dien mon an
├── main.ipynb      # Notebook tai dataset va train model
└── model.keras     # Model Keras da train
```

## Cach chay ung dung

Mo terminal tai thu muc `VIETNAMESE DISHES`, sau do chay:

```bash
python gui.py
```

Trong giao dien:

- Chon `Mo webcam realtime` de nhan dien mon an bang webcam.
- Nhan `C` de chup/chot ket qua hien tai.
- Nhan `R` de restart/quay lai che do quet.
- Nhan `Q` hoac `Esc` de thoat.
- Chon `Import anh mon an` de nhan dien anh co san trong may.

## Cach hoat dong

`gui.py` load model tu:

```text
model.keras
```

Anh dau vao duoc xu ly nhu sau:

1. Resize ve `128 x 128`.
2. Chuyen thanh mang numpy.
3. Chuan hoa gia tri pixel ve khoang `0..1`.
4. Dua qua model Keras.
5. Lay lop co xac suat cao nhat va hien thi ten mon + do tin cay.

## Train lai model

Mo file:

```text
main.ipynb
```

Notebook hien tai tai dataset tu KaggleHub:

```python
kagglehub.dataset_download("quandang/vietnamese-foods")
```

Dataset duoc cau hinh luu trong:

```text
D:\KAGGLE_DATASETS
```

Trong notebook, cac lop duoc chon de train la:

```python
selected_classes = [
    "Banh cuon",
    "Banh mi",
    "Banh xeo",
    "Bun dau mam tom",
    "Banh canh"
]
```

Model duoc train voi:

- kich thuoc anh: `128 x 128`
- batch size: `64`
- so epoch toi da: `80`
- checkpoint luu model tot nhat theo `val_accuracy`
- file output: `model.keras`

Sau khi train lai, `model.keras` se duoc cap nhat va `gui.py` se dung model moi nay.

## File quan trong

Can giu neu chi muon chay ung dung:

- `gui.py`
- `model.keras`

Can giu neu muon train lai:

- `main.ipynb`
- `model.keras` neu muon giu model hien tai
- dataset Kaggle trong `D:\KAGGLE_DATASETS` neu da tai ve

## Co the don dep

Khong nen xoa neu con muon chay app:

- `gui.py`
- `model.keras`

Khong nen xoa neu con muon train lai:

- `main.ipynb`
- dataset da tai trong `D:\KAGGLE_DATASETS`

## Luu y khi sua lop mon an

Neu muon them/bot mon an, can sua dong bo 2 noi:

1. `selected_classes` trong `main.ipynb`
2. `class_labels` va `class_labels_no_accent` trong `gui.py`

Thu tu lop trong `gui.py` phai khop voi thu tu class khi train. Neu thu tu bi lech, model co the du doan dung index nhung hien sai ten mon.