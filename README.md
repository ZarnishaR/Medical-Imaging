# Medical Imaging 

YOLOv8-based medical imaging application for classification, detection, and segmentation tasks.

## Tasks

1. **Classification:** COVID-19, Viral Pneumonia, and Normal lung images
2. **Object Detection:** RBC, WBC, and Platelets detection
3. **Segmentation:** Breast ultrasound image segmentation

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd YOLOv8-Medical-Imaging
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

Demo images are provided in the `DEMO_IMAGES` directory. 