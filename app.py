from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from database.db import init_db
from database.seed import seed_vehicle_data, seed_violation_data
from models.plate_detector import detect_number_plate, draw_plate_box
from models.plate_ocr import extract_plate_text
from models.violation_detector import detect_violation_from_full_image
from services.challan_service import fetch_recent_logs, get_owner_by_plate, get_violation_amount, match_violation_in_db
from services.whatsapp_service import generate_whatsapp_message, save_challan_log, send_whatsapp_message
from utils.image_utils import read_image_from_upload, safe_crop, to_rgb
from utils.text_utils import normalize_plate

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Traffic Sentinel: AI Violation Detection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Base Font & Theme */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* App Background (Dark Gradient) */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0c1017 0%, #151b29 100%);
    color: #e2e8f0;
}

/* Hide Default Streamlit Menus & Watermarks */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sexy Title Gradient */
h1 {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: -1px;
}

/* Sidebar Customization (Glassmorphism) */
[data-testid="stSidebar"] {
    background-color: rgba(13, 17, 23, 0.85) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #00F2FE !important;
}

/* Main Action Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem;
    padding: 10px 0;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 65, 108, 0.5);
    background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
}

/* Metrics Cards (Glass effect + glowing texts) */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 242, 254, 0.3);
}

[data-testid="stMetricValue"] {
    color: #00F2FE !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* File Uploader Customization */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px dashed rgba(0, 242, 254, 0.4);
    border-radius: 12px;
    padding: 10px;
    transition: all 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00F2FE;
    background: rgba(0, 242, 254, 0.05);
}

/* Images Padding */
[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
/* Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: none;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
    color: #0c1017 !important;
    font-weight: 800 !important;
    border: none;
}
</style>
""", unsafe_allow_html=True)
def setup_database() -> bool:
    init_db()
    seed_vehicle_data(force_reset=True)
    seed_violation_data(force_reset=True)
    return True


setup_database()

st.title("🚨 Traffic Sentinel: AI Violation Detection")
st.caption("Automatic flow only: upload an image, detect plate, classify vehicle + violation with Grok, match database, and issue challan.")

with st.sidebar:
    st.header("Project Rules")
    st.markdown(
        """
- Only one owner exists in database:
  - **Sanjana Kanaki**
  - **9067443576**
  - **MH13 AB1234**
- **Bike:** overspeed, triple riding
- **Car:** overspeed only
- **Truck:** overspeed only
- Manual plate entry removed
        """
    )
    st.info("To send a real WhatsApp message, set WHATSAPP_MODE=twilio and add Twilio WhatsApp credentials in .env.")

tab_detect, tab_dashboard, tab_registry = st.tabs(["🚦 Live Detection", "📊 System Dashboard", "📜 Challan Registry"])

with tab_detect:
    st.markdown("### 📸 Upload Traffic Evidence")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG) to analyze", type=["jpg", "jpeg", "png"])
    run_button = st.button("Run Automatic Challan Generation", type="primary")
    show_debug = st.checkbox("Show debug details", value=False)

    if uploaded_file is not None:
        image_bgr = read_image_from_upload(uploaded_file)
        st.subheader("Evidence Capture")
        st.image(to_rgb(image_bgr), width="stretch")

        if run_button:
            with st.spinner("Analyzing image..."):
                bbox = detect_number_plate(image_bgr)
                annotated = draw_plate_box(image_bgr, bbox)
                cropped_plate = None
                ocr_result = {"raw_text": "", "plate_number": ""}

                if bbox is not None:
                    cropped_plate = safe_crop(image_bgr, bbox)
                    ocr_result = extract_plate_text(cropped_plate)

                extracted_plate = normalize_plate(ocr_result.get("plate_number", ""))

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=BASE_DIR / "temp") as temp_file:
                    temp_image_path = temp_file.name
                    cv2.imwrite(temp_image_path, image_bgr)

                owner = get_owner_by_plate(extracted_plate) if extracted_plate else None
                
                known_vehicle_type = "unknown"
                if owner and extracted_plate:
                    if normalize_plate(owner.get("four_wheeler_number")) == extracted_plate:
                        known_vehicle_type = "car"
                    elif normalize_plate(owner.get("two_wheeler_number")) == extracted_plate:
                        known_vehicle_type = "bike"

                violation_result = detect_violation_from_full_image(temp_image_path, known_vehicle_type)
                vehicle_type = violation_result.get("vehicle_type", "unknown")
                detected_violation = violation_result.get("violation_name", "unknown")
                reason = violation_result.get("reason", "")

                # Auto-register if not found
                if owner is None and extracted_plate and vehicle_type in ["car", "bike", "truck"]:
                    from database.db import execute_query
                    four_wheeler = extracted_plate if vehicle_type in ["car", "truck"] else None
                    two_wheeler = extracted_plate if vehicle_type == "bike" else None
                    execute_query(
                        """
                        INSERT INTO vehicle_database
                        (owner_name, mobile_number, four_wheeler_number, two_wheeler_number, location)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        ("Rahul Singh", "9067443576", four_wheeler, two_wheeler, "Unknown Location")
                    )
                    owner = get_owner_by_plate(extracted_plate)

                violation_row = match_violation_in_db(vehicle_type, detected_violation)
                amount = get_violation_amount(vehicle_type, detected_violation) if violation_row else None

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Plate Detection")
                st.image(to_rgb(annotated), width="stretch")
                if bbox is None:
                    st.warning("Automatic plate detector did not find a plate region.")
                else:
                    st.success("Plate region detected.")

            with col2:
                st.subheader("Cropped Plate")
                if cropped_plate is not None:
                    st.image(to_rgb(cropped_plate), width="stretch")
                else:
                    st.info("No cropped plate available.")

            st.subheader("Extraction Result")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Plate", extracted_plate or "Not Found")
            m2.metric("Vehicle Type", vehicle_type.title())
            m3.metric("Violation", detected_violation.title())
            m4.metric("Amount", f"₹{amount}" if amount is not None else "Not Available")

            if show_debug:
                with st.expander("Debug / Detection Details"):
                    st.json(
                        {
                            "ocr_raw_text": ocr_result.get("raw_text", ""),
                            "plate_normalized": extracted_plate,
                            "vehicle_type": vehicle_type,
                            "violation": detected_violation,
                            "reason": reason,
                        }
                    )

            st.subheader("Owner Details")
            if owner:
                st.success("Owner found in database.")
                st.json(
                    {
                        "owner_name": owner["owner_name"],
                        "mobile_number": owner["mobile_number"],
                        "number_plate": extracted_plate,
                        "location": owner["location"],
                    }
                )
            else:
                st.error("Owner not found in database.")

            can_issue = bool(owner and extracted_plate and amount is not None and detected_violation != "unknown")
            if can_issue:
                message = generate_whatsapp_message(
                    owner=owner["owner_name"],
                    plate=extracted_plate,
                    violation=detected_violation,
                    amount=amount,
                    location=owner["location"],
                )
                st.subheader("WhatsApp Preview")
                st.text_area("Generated message", message, height=180)

                # Generate WhatsApp deep link
                import urllib.parse
                
                # Format phone number for WhatsApp API
                phone = owner["mobile_number"].strip()
                if not phone.startswith("+") and not phone.startswith("91") and len(phone) == 10:
                    phone = f"91{phone}"
                    
                whatsapp_url = f"https://api.whatsapp.com/send?phone={phone}&text={urllib.parse.quote(message)}"

                st.markdown(
                    f'''<a href="{whatsapp_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #25D366; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">📲 Send via WhatsApp</a>''',
                    unsafe_allow_html=True
                )

                # Log to digital registry automatically
                save_challan_log(
                    plate_number=extracted_plate,
                    owner_name=owner["owner_name"],
                    mobile_number=owner["mobile_number"],
                    vehicle_type=vehicle_type,
                    violation=detected_violation,
                    amount=amount,
                    message=message,
                    status="demo_link_generated",
                )
                
                st.success("Challan prepared! Click the WhatsApp button above to send the alert.")
            else:
                st.info("Automatic challan could not be issued because one or more detection steps failed.")

with tab_dashboard:
    st.markdown("### 📈 System Overview & Analytics")
    from database.db import fetch_one
    
    # Safely fetch stats (handling cases where tables might be empty)
    try:
        total_challans = fetch_one("SELECT COUNT(*) as count FROM challan_logs")["count"]
        total_revenue = fetch_one("SELECT SUM(amount) as total FROM challan_logs")["total"] or 0
        total_vehicles = fetch_one("SELECT COUNT(*) as count FROM vehicle_database")["count"]
    except Exception:
        total_challans, total_revenue, total_vehicles = 0, 0, 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Violations Caught", f"{total_challans:,}")
    col2.metric("💰 Total Fine Revenue", f"₹{total_revenue:,}")
    col3.metric("🚗 Registered Vehicles", f"{total_vehicles:,}")
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("#### About the Traffic Sentinel Engine")
    st.info("The dashboard updates in real-time as new traffic violations are automatically detected and logged by the AI.")

with tab_registry:
    st.markdown("### 📜 Digital Challan Registry")
    st.caption("A historical log of all detected traffic violations and generated challans.")
    logs = fetch_recent_logs()
    if logs:
        st.dataframe(logs, width="stretch", height=500)
    else:
        st.write("No challan logs yet. Run a detection to generate logs.")
