# 🍃 Viridis — Eco-Friendly Hospital Design Studio

An automated, parametric design platform for sustainable healthcare facilities. Generates compliant 2D blueprints, 3D renderings, engineering analyses, and green-building certification forecasts for individual departments or full hospital campuses.

**Built as a graduation project (Biomedical Engineering, 2025).**

---

## ✨ Features

### 🏥 Two Design Modes
- **Single Department:** detailed room-level design with equipment placement, safety systems, and material specifications.
- **Full Hospital Site Plan:** campus-scale design with up to 24 departments, healing gardens, solar farms, and complete infrastructure.

### 📐 Outputs Generated
- 2D blueprint with measurements (FGI 2018 / Egyptian MoH compliant)
- Interactive 3D rendering (Plotly)
- HVAC load calculations (ASHRAE 170)
- Fire safety analysis with **animated evacuation simulation** (NFPA SFPE method)
- Water flow Sankey diagram + Power single-line diagram (NFPA 99)
- Bill of Quantities (CSV + Excel export)
- Functional Program document (Markdown + Word export)
- **LEED / Estidama / Mostadam green-building certification forecast**

### 🚨 Safety & Emergency Systems (16 features)
- Smoke detectors, sprinklers, pull stations, emergency lighting (NFPA 13/72/101)
- AED stations, eye-wash, safety showers, patient call buttons (AHA, ANSI Z358.1)
- Egress paths, wayfinding signs (ISO 7010), ADA-compliant refuge areas
- Radiation, isolation, and hazmat markers (NCRP 49, CDC, OSHA HCS)
- 🎬 **Animated fire evacuation simulation** with NFPA-compliant timing

### 🌱 Sustainability Materials Database
22 eco-friendly materials across 5 categories (flooring, wall, ceiling, insulation, paint) with full specifications, hospital use-cases, and certification references.

### 🤖 Smart Features
- AI design checks & rule-based chatbot
- Cost optimizer (target: budget / CO₂ / LEED)
- 26-city climate database with solar + rainwater potential
- Budget manager with anomaly detection

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run viridis_v24.py
```

The app opens at `http://localhost:8501`.

---

## 📚 Standards & References

- **FGI 2018** — Facility Guidelines Institute (US healthcare design)
- **Egyptian MoH** — Ministry of Health construction standards
- **ASHRAE 170 / 189.3** — Healthcare ventilation + green building
- **NFPA 13/72/101/704/99** — Fire protection codes
- **AHA / ANSI Z358.1 / ADA 4.3 / ISO 7010 / NCRP 49 / OSHA HCS 2012**
- **LEED v4 Healthcare / Estidama / Mostadam** — Green certifications

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **2D rendering:** Matplotlib
- **3D rendering:** Plotly (interactive)
- **Numerics:** NumPy
- **Exports:** Pandas, python-docx, openpyxl, PIL

---

## 📄 License

Educational use only — graduation project.

## 👤 Author

Biomedical Engineering · Graduation Project 2025
