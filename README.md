# Traffic Sentinel: AI Violation Detection

This rebuilt project is simplified around your exact demo flow:

- Only one owner in the database: **Sanjana Kanaki**
- Mobile: **9067443576**
- Plate: **MH13 AB1234**
- Automatic plate OCR from the image
- Automatic vehicle + violation classification using Grok
- Automatic challan generation
- Optional real WhatsApp sending through Twilio WhatsApp if credentials are configured

## Supported rules

- **Bike** -> `overspeed`, `triple riding`
- **Car** -> `overspeed`
- **Truck** -> `overspeed`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `XAI_API_KEY`.

## Run

```bash
streamlit run app.py
```

## Notes

- The app removes manual plate entry.
- If Grok OCR reads `MH13AB1234` from the uploaded image, the owner will match automatically.
- If you want real WhatsApp delivery, set `WHATSAPP_MODE=twilio` and add Twilio credentials.
- If Twilio credentials are not present, the app still creates the challan and saves a demo send log.
