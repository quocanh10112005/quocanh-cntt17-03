import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="OCR Hóa đơn", layout="wide")
st.title("📄 OCR Hóa đơn - Trích xuất, Sửa & Lưu dữ liệu")

tab1, tab2 = st.tabs(["📂 Upload ảnh", "📷 Quét từ Camera"])

# ---------- Tiền xử lý ảnh ----------
def preprocess(img_rgb, mode="clahe_otsu"):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    max_side = max(h, w)
    if max_side < 1800:
        scale = 1800 / max_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    if mode == "clahe_otsu":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        bin_img = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif mode == "adaptive":
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )
    else:
        bin_img = gray

    return bin_img

# ---------- OCR ----------
def run_ocr(image_bin, psm):
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(image_bin, lang="vie+eng", config=config)
    data = pytesseract.image_to_data(image_bin, lang="vie+eng", config=config, output_type=pytesseract.Output.DATAFRAME)
    if "conf" in data.columns:
        confs = data["conf"]
        confs = confs[confs.astype(str) != "-1"]
        mean_conf = float(confs.astype(float).mean()) if len(confs) else -1.0
    else:
        mean_conf = -1.0
    return text, mean_conf

# ---------- Trích xuất thông tin ----------
def extract_info(text):
    # --- Số hóa đơn ---
    patterns_hd = [
        r'hóa\s*đơn\s*#\s*(\d+)',
        r'hóa\s*đơn\s*số\s*[:\s]*(\d+)',
        r'hóa\s*đơn\s*[:\s]*(\d+)',
        r'#\s*(\d+)',
        r'số\s*[:\s]*(\d+)',
        r'H[ĐD]?\s*(\d+)',
        r'Invoice\s*#\s*(\d+)'
    ]
    so_hd = None
    for p in patterns_hd:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            so_hd = m.group(1)
            break

    # --- Ngày ---
    patterns_ngay = [
        r'ngày\s+(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})',
        r'(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[.]\d{1,2}[.]\d{2,4})',
        r'ngày\s*[:]\s*(\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{4})'
    ]
    ngay = None
    for p in patterns_ngay:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            date_str = m.group(1)
            if 'tháng' in date_str and 'năm' in date_str:
                parts = re.findall(r'\d+', date_str)
                if len(parts) >= 3:
                    day, month, year = parts[0], parts[1], parts[2]
                    ngay = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
            else:
                ngay = date_str
            break

    # --- Tổng tiền ---
    patterns_tong = [
        r'Tổng\s*thanh\s*toán\s*[:\s]*([\d.,]+)\s*[đĐkK]?',
        r'Tổng\s*cộng\s*[:\s]*([\d.,]+)\s*[đĐkK]?',
        r'Tiền\s*thanh\s*toán\s*[:\s]*([\d.,]+)\s*[đĐkK]?'
    ]
    tong_tien = None
    for pattern in patterns_tong:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            number_str = match.group(1)
            clean_number = re.sub(r'[^\d]', '', number_str)
            full_match = match.group(0).lower()
            try:
                amount = float(clean_number)
                if 'k' in full_match and 'đ' not in full_match:
                    amount *= 1000
                tong_tien = f"{amount:,.0f}đ".replace(",", ".")
                break
            except:
                continue

    # --- Địa chỉ ---
    patterns_dia_chi = [
        r'địa\s*chỉ\s*khách\s*hàng\s*[:\s]*([^\n\r]{10,100})',
        r'(\d+\s+điện\s+biên[^\n\r]*)',
        r'(\d+\s+đường\s+[a-zA-Z0-9\s]+[^\n\r]*)',
        r'(số\s+\d+[^\n\r]{10,80})'
    ]
    dia_chi = None
    for p in patterns_dia_chi:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            dia_chi = m.group(1).strip()
            break

    return {
        "Số hóa đơn": so_hd if so_hd else "",
        "Ngày": ngay if ngay else "",
        "Tổng tiền": tong_tien if tong_tien else "",
        "Địa chỉ": dia_chi if dia_chi else ""
    }

# ---------- Xử lý ảnh ----------
def process_image(pil_img):
    pil = ImageOps.exif_transpose(pil_img).convert("RGB")
    img_rgb = np.array(pil)

    bin1 = preprocess(img_rgb, mode="clahe_otsu")
    bin2 = preprocess(img_rgb, mode="adaptive")

    candidates = []
    for bin_img, prep_name in [(bin1, "CLAHE+Otsu"), (bin2, "Adaptive")]:
        for psm in [6, 4, 11]:
            txt, score = run_ocr(bin_img, psm)
            candidates.append({
                "prep": prep_name,
                "psm": psm,
                "text": txt,
                "score": score,
                "img": bin_img
            })
    best = max(candidates, key=lambda x: x["score"])
    return pil, best

# ---------- Hiển thị ----------
def show_results(pil, best):
    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader("🖼 Ảnh hóa đơn & ảnh xử lý")
        st.image(pil, caption="Ảnh gốc", use_container_width=True)
        st.image(best["img"], caption=f"Xử lý tốt nhất: {best['prep']} | PSM {best['psm']}", use_container_width=True, channels="GRAY")
    with col2:
        st.subheader("📑 Văn bản OCR")
        st.caption(f"Chế độ: {best['prep']} | PSM {best['psm']} | Độ tin cậy TB: {round(best['score'],2)}")

        # --- Vùng chỉnh sửa text OCR ---
        ocr_text = st.text_area("Kết quả OCR (có thể sửa)", value=best["text"], height=300)

        # --- Nút tải văn bản OCR ---
        st.download_button(
            "⬇️ Tải văn bản OCR",
            ocr_text,
            file_name="van_ban_ocr.txt",
            mime="text/plain"
        )
        st.markdown("---")
        
        # --- Trích xuất thông tin ---
        info = extract_info(ocr_text)

        st.subheader("📋 Thông tin trích xuất (có thể sửa)")
        # Cho phép sửa từng trường
        info["Số hóa đơn"] = st.text_input("Số hóa đơn", value=info["Số hóa đơn"])
        info["Ngày"] = st.text_input("Ngày", value=info["Ngày"])
        info["Tổng tiền"] = st.text_input("Tổng tiền", value=info["Tổng tiền"])
        info["Địa chỉ"] = st.text_input("Địa chỉ", value=info["Địa chỉ"])

        # --- Xuất dữ liệu ---
        df_out = pd.DataFrame([info])

        # CSV
        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button("⬇️ Tải CSV", data=csv_buffer.getvalue(), file_name="hoadon_ocr.csv", mime="text/csv")

        # Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="OCR")
        excel_buffer.seek(0)
        st.download_button("⬇️ Tải Excel", data=excel_buffer.getvalue(), file_name="hoadon_ocr.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============ Upload ảnh ============
with tab1:
    uploaded_file = st.file_uploader("Tải ảnh hóa đơn lên", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        pil = Image.open(uploaded_file)
        pil, best = process_image(pil)
        show_results(pil, best)

# ============ Quét Camera ============
with tab2:
    cam_file = st.camera_input("Chụp ảnh hóa đơn bằng camera")
    if cam_file is not None:
        pil = Image.open(cam_file)
        pil, best = process_image(pil)
        show_results(pil, best)