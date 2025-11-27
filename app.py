import os
import sys
import uuid
from datetime import datetime
import logging
import importlib.util
import re
import json

from flask import Flask, request, jsonify, send_from_directory, render_template
import io
import requests

PREDICT_IMPORT_ERROR = None  # populated if import fails


def _safe_exec_module(spec):
    global PREDICT_IMPORT_ERROR
    try:
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    except Exception as e:
        PREDICT_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("Failed to import prediction utilities")
    return None


def _load_predict_utils(app_root: str):
    """Try importing predict utilities from several common locations.

    Order:
    1) package import: from utils import predict
    2) file at app_root/utils/predict.py
    3) file at app_root/predict.py
    4) package import: from brain_tumor_api import predict
    5) file at app_root/brain_tumor_api/predict.py
    6) package import: from brain_tumor_api.utils import predict
    7) file at app_root/brain_tumor_api/utils/predict.py
    """
    # 1) regular package import
    try:
        from utils import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from utils import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"utils package import failed: {e}")

    # 2) direct file import utils/predict.py
    utils_predict_path = os.path.join(app_root, "utils", "predict.py")
    if os.path.exists(utils_predict_path):
        spec = importlib.util.spec_from_file_location("utils.predict", utils_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {utils_predict_path}")
            return predict_mod

    # 3) direct file import predict.py in root
    root_predict_path = os.path.join(app_root, "predict.py")
    if os.path.exists(root_predict_path):
        spec = importlib.util.spec_from_file_location("predict", root_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {root_predict_path}")
            return predict_mod

    # 4) package import: brain_tumor_api.predict
    try:
        from brain_tumor_api import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from brain_tumor_api import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"brain_tumor_api package import failed: {e}")

    # 5) direct file import brain_tumor_api/predict.py
    bta_predict_path = os.path.join(app_root, "brain_tumor_api", "predict.py")
    if os.path.exists(bta_predict_path):
        spec = importlib.util.spec_from_file_location("brain_tumor_api.predict", bta_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {bta_predict_path}")
            return predict_mod

    # 6) package import: brain_tumor_api.utils.predict
    try:
        from brain_tumor_api.utils import predict as predict_mod  # type: ignore
        logging.info("Loaded predict utils via package import 'from brain_tumor_api.utils import predict'")
        return predict_mod
    except Exception as e:
        logging.debug(f"brain_tumor_api.utils package import failed: {e}")

    # 7) direct file import brain_tumor_api/utils/predict.py
    bta_utils_predict_path = os.path.join(app_root, "brain_tumor_api", "utils", "predict.py")
    if os.path.exists(bta_utils_predict_path):
        spec = importlib.util.spec_from_file_location("brain_tumor_api.utils.predict", bta_utils_predict_path)
        predict_mod = _safe_exec_module(spec)
        if predict_mod is not None:
            logging.info(f"Loaded predict utils from {bta_utils_predict_path}")
            return predict_mod

    logging.error("Could not locate a predict.py module. Place your inference code under utils/, brain_tumor_api/, or root.")
    return None


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "static")
GRADCAM_DIR = os.path.join(STATIC_DIR, "gradcam")
REPORTS_DIR = os.path.join(STATIC_DIR, "reports")
DATA_DIR = os.path.join(APP_ROOT, "data")
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
logger = logging.getLogger(__name__)

# Load predict utils once at startup
predict_utils = _load_predict_utils(APP_ROOT)


def build_treatment_suggestion(label: str) -> str:
    mapping = {
        "glioma": (
            "Biopsy for histologic and molecular profiling; maximal safe surgical resection; "
            "adjuvant radiotherapy and/or temozolomide based on grade and markers (e.g., IDH, 1p/19q); "
            "neuro‑oncology follow‑up with guideline‑based MRI surveillance."
        ),
        "meningioma": (
            "Surgical resection when feasible; consider stereotactic radiosurgery for small, skull‑base, or residual tumors; "
            "postoperative MRI at 3–6 months, then every 6–12 months depending on WHO grade and symptoms."
        ),
        "pituitary": (
            "Transsphenoidal resection for symptomatic adenomas; endocrinology management and targeted medical therapy "
            "(e.g., dopamine agonists for prolactinoma); adjuvant radiotherapy for residual/recurrent disease; "
            "MRI and hormone follow‑up every 3–6 months initially."
        ),
        "no_tumor": (
            "No tumor detected. Correlate with clinical findings; routine follow‑up or repeat imaging only if symptoms persist."
        ),
    }
    return mapping.get(label, "Consult a specialist for a tailored treatment plan")


def human_readable_diagnostic(label: str, confidence: float) -> str:
    if label == "no_tumor":
        return "The model did not detect a brain tumor in this MRI."
    return f"The model suggests a {label} with a confidence of {confidence:.1%}. Consider clinical correlation and further evaluation."


def estimate_lobe_from_gradcam(gradcam_path: str) -> str:
    """Coarse lobe + hemisphere estimate directly from the Grad-CAM overlay image.

    Uses the most activated pixel (argmax) from a color overlay as a proxy.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(gradcam_path).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        # Use red channel as proxy for activation (typical hot colormap), fall back to blue if needed
        red = arr[..., 0]
        if red.max() <= 0.0:
            red = arr[..., 2]
        y, x = np.unravel_index(np.argmax(red), red.shape)
        h, w = red.shape

        hemisphere = "left" if x < w / 2 else "right"

        y_rel = y / max(h, 1)
        # Split into thirds vertically to approximate lobes
        if y_rel < 0.33:
            lobe = "frontal or parietal (superior)"
        elif y_rel < 0.66:
            lobe = "parietal or temporal (mid)"
        else:
            lobe = "temporal or occipital (inferior/posterior)"

        return f"{lobe}, {hemisphere} hemisphere"
    except Exception:
        return "undetermined"


def build_highlight_tokens(label: str, localization: str) -> list[dict]:
    tokens: list[dict] = []
    if label:
        tokens.append({"text": label, "type": "label"})
    if localization and localization != "undetermined":
        for word in ["frontal", "parietal", "temporal", "occipital", "left", "right"]:
            if word in localization.lower():
                tokens.append({"text": word, "type": "anat"})
    proc_by_label = {
        "glioma": ["biopsy", "resection", "radiotherapy", "temozolomide"],
        "meningioma": ["resection", "stereotactic radiosurgery"],
        "pituitary": ["transsphenoidal resection", "endocrinology", "radiotherapy"],
    }
    for t in proc_by_label.get(label, []):
        tokens.append({"text": t, "type": "proc"})
    return tokens


def try_predict_with_utils(image_path: str, class_names: list[str]):
    """Attempt to call into utils.predict using common function names.

    Expected returns: (pred_label: str, confidences: dict[str, float], gradcam_np_or_path)
    """
    if predict_utils is None:
        raise RuntimeError("utils/predict.py not importable. Ensure it exists and is importable.")

    # Flexible discovery of available functions
    # Priority 1: a single entrypoint that returns everything
    for fn_name in [
        "predict_and_gradcam",
        "run_inference",
        "predict_full",
    ]:
        if hasattr(predict_utils, fn_name):
            fn = getattr(predict_utils, fn_name)
            logger.info(f"Calling utils.{fn_name}(...) with image_path={os.path.basename(image_path)}")
            # Try with common kwargs; fall back to positional
            try:
                result = fn(image_path=image_path, class_names=class_names, output_dir=GRADCAM_DIR)  # type: ignore
            except TypeError:
                result = fn(image_path, class_names, GRADCAM_DIR)  # type: ignore
            # Expect tuple
            return result

    # Priority 2: separate prediction and gradcam generators
    pred_fn = None
    for name in ["predict_image", "predict", "classify_image"]:
        if hasattr(predict_utils, name):
            pred_fn = getattr(predict_utils, name)
            break

    cam_fn = None
    for name in ["generate_gradcam", "gradcam", "make_gradcam", "compute_gradcam"]:
        if hasattr(predict_utils, name):
            cam_fn = getattr(predict_utils, name)
            break

    if pred_fn is None or cam_fn is None:
        raise RuntimeError(
            "predict module must expose either predict_and_gradcam(...) or both predict/predict_image(...) and generate_gradcam(...)."
        )

    logger.info("Calling classification function from utils …")
    # Try common call signatures
    try:
        pred_label, confidences = pred_fn(image_path=image_path, class_names=class_names)  # type: ignore
    except TypeError:
        # Some modules accept only image_path
        try:
            pred_label, confidences = pred_fn(image_path)  # type: ignore
        except TypeError:
            pred_label, confidences = pred_fn(image_path, class_names)  # type: ignore
    logger.info("Calling Grad-CAM generator from utils …")
    try:
        gradcam_obj = cam_fn(image_path=image_path, output_dir=GRADCAM_DIR)  # type: ignore
    except TypeError:
        try:
            gradcam_obj = cam_fn(image_path, GRADCAM_DIR)  # type: ignore
        except TypeError:
            gradcam_obj = cam_fn(image_path)  # type: ignore
    return pred_label, confidences, gradcam_obj


def ensure_gradcam_path(gradcam_obj) -> str:
    """Accepts a numpy array/PIL.Image/str path and ensures it becomes a saved file path under static/gradcam."""
    try:
        # If it's already a path
        if isinstance(gradcam_obj, str) and os.path.exists(gradcam_obj):
            return gradcam_obj

        # Try PIL Image save
        from PIL import Image  # lazy import
        if isinstance(gradcam_obj, Image.Image):
            filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(GRADCAM_DIR, filename)
            gradcam_obj.save(out_path)
            return out_path

        # Try numpy array -> save via PIL
        import numpy as np
        if isinstance(gradcam_obj, np.ndarray):
            from PIL import Image  # type: ignore
            img = Image.fromarray(gradcam_obj)
            filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(GRADCAM_DIR, filename)
            img.save(out_path)
            return out_path
    except Exception:
        pass

    # Fallback: write bytes-like object if possible
    filename = f"gradcam_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(GRADCAM_DIR, filename)
    try:
        with open(out_path, "wb") as f:
            f.write(gradcam_obj)
        return out_path
    except Exception:
        # As a last resort, return under static even if not existent; frontend will handle error
        return out_path


def normalize_confidences(confidences, class_names: list[str]):
    """Convert confidences to a JSON-serializable dict[label -> float].

    Accepts: dict/Mapping label->score, list/tuple aligned with class_names,
    or numpy array. Returns dict with native Python floats.
    """
    try:
        import numpy as np  # local import
    except Exception:  # pragma: no cover
        np = None

    # If dict-like already
    if isinstance(confidences, dict):
        normalized = {}
        for k, v in confidences.items():
            try:
                normalized[str(k)] = float(v)
            except Exception:
                try:
                    normalized[str(k)] = float(v.item())  # type: ignore[attr-defined]
                except Exception:
                    normalized[str(k)] = 0.0
        return normalized

    # If it's a numpy array or list-like aligned to class_names
    values = confidences
    if np is not None and hasattr(np, "ndarray") and isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        out = {}
        for idx, label in enumerate(class_names):
            try:
                out[label] = float(values[idx])
            except Exception:
                out[label] = 0.0
        return out

    # Fallback: try to coerce to float and attach to predicted label later
    return {label: 0.0 for label in class_names}


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # Class names as provided
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

    # Optional: preload model to reduce first-request latency, if utils exposes it
    model_handle = None
    if predict_utils is not None:
        for name in ["load_model", "get_model", "init_model"]:
            if hasattr(predict_utils, name):
                try:
                    loader = getattr(predict_utils, name)
                    logger.info(f"Preloading model via utils.{name}() …")
                    model_handle = loader()
                    logger.info("Model preloaded successfully")
                except Exception:
                    logger.exception("Model preload failed; proceeding without preloaded model")
                    model_handle = None
                break

    @app.route("/")
    def serve_index():
        return send_from_directory(APP_ROOT, "index.html")

    @app.route("/analysis")
    def serve_analysis():
        return send_from_directory(APP_ROOT, "analysis.html")
        
    # Load patients from persistent storage
    def load_patients():
        """Load patients from JSON file"""
        try:
            if os.path.exists(PATIENTS_FILE):
                with open(PATIENTS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading patients: {e}")
        return []
    
    def save_patients(patients_data):
        """Save patients to JSON file"""
        try:
            with open(PATIENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(patients_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(patients_data)} patients to {PATIENTS_FILE}")
        except Exception as e:
            logger.error(f"Error saving patients: {e}")
    
    # Load patients on startup
    patients = load_patients()
    logger.info(f"Loaded {len(patients)} patients from storage")
    
    # In-memory storage for uploaded reports (in a real app, use a database)
    uploaded_reports = {}  # {report_id: {filename, path, upload_time, metadata}}
    
    @app.route("/dashboard")
    def serve_dashboard():
        return render_template("index.html")
    
    @app.route("/api/patients", methods=["GET"])
    def get_patients():
        return jsonify(patients)
    
    @app.route("/api/patients", methods=["POST"])
    def add_patient():
        data = request.get_json()
        new_patient = {
            "id": str(uuid.uuid4())[:8],
            "name": data.get("name", ""),
            "age": data.get("age", ""),
            "condition": data.get("condition", ""),
            "lastVisit": datetime.utcnow().strftime("%Y-%m-%d"),
            "report": data.get("report", ""),
            "imagePath": data.get("imagePath", ""),
            "status": data.get("status", "active"),
            "phone": data.get("phone", ""),
            "created_at": datetime.utcnow().isoformat()
        }
        patients.append(new_patient)
        save_patients(patients)  # Save to file
        return jsonify({"status": "success", "patient": new_patient})
    
    @app.route("/api/patients/<patient_id>", methods=["DELETE"])
    def delete_patient(patient_id):
        """Delete a patient"""
        global patients
        patients = [p for p in patients if p.get("id") != patient_id]
        save_patients(patients)  # Save to file
        return jsonify({"status": "success"})
    
    @app.route("/api/search", methods=["GET"])
    def search():
        """Search patients and reports by name"""
        query = request.args.get("q", "").lower().strip()
        if not query:
            return jsonify({"patients": [], "reports": []})
        
        # Search patients by name
        matching_patients = []
        for patient in patients:
            patient_name = patient.get("name", "").lower()
            patient_id = patient.get("id", "").lower()
            if query in patient_name or query in patient_id:
                matching_patients.append(patient)
        
        # Search reports by filename and patient name
        matching_reports = []
        try:
            if os.path.exists(REPORTS_DIR):
                for filename in os.listdir(REPORTS_DIR):
                    if filename.endswith('.pdf'):
                        # Check if filename matches
                        if query in filename.lower():
                            file_path = os.path.join(REPORTS_DIR, filename)
                            file_stat = os.stat(file_path)
                            
                            # Try to find patient by matching name in filename or by report URL
                            patient_match = None
                            for patient in patients:
                                patient_name_lower = patient.get("name", "").lower()
                                if patient_name_lower and patient_name_lower in query:
                                    # Check if this patient has this report
                                    patient_report = patient.get("report", "")
                                    if filename in patient_report or patient_name_lower in filename.lower():
                                        patient_match = patient
                                        break
                            
                            rel_path = os.path.relpath(file_path, STATIC_DIR).replace("\\", "/")
                            report_info = {
                                "filename": filename,
                                "url": f"/static/{rel_path}",
                                "date": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                                "size": file_stat.st_size,
                                "patient_name": patient_match.get("name", "") if patient_match else ""
                            }
                            matching_reports.append(report_info)
        except Exception as e:
            logger.error(f"Error searching reports: {e}")
        
        return jsonify({
            "patients": matching_patients,
            "reports": matching_reports,
            "query": query
        })
    
    @app.route("/api/reports", methods=["GET"])
    def get_reports():
        """Get list of all PDF reports"""
        try:
            reports = []
            if os.path.exists(REPORTS_DIR):
                for filename in sorted(os.listdir(REPORTS_DIR), reverse=True):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(REPORTS_DIR, filename)
                        file_stat = os.stat(file_path)
                        # Extract date from filename: report_20251005T121222_5fd9bf.pdf
                        try:
                            date_str = filename.split('_')[1] if '_' in filename else ""
                            if 'T' in date_str:
                                report_date = datetime.strptime(date_str, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M")
                            else:
                                report_date = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                        except:
                            report_date = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                        
                        rel_path = os.path.relpath(file_path, STATIC_DIR).replace("\\", "/")
                        reports.append({
                            "filename": filename,
                            "url": f"/static/{rel_path}",
                            "date": report_date,
                            "size": file_stat.st_size
                        })
            return jsonify(reports)
        except Exception as e:
            logger.exception("Error fetching reports")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/patient/<patient_id>", methods=["GET"])
    def get_patient_details(patient_id):
        """Get detailed information about a specific patient"""
        patient = next((p for p in patients if p.get("id") == patient_id), None)
        if patient:
            return jsonify(patient)
        return jsonify({"error": "Patient not found"}), 404
    
    @app.route("/chat")
    def serve_chat():
        return send_from_directory(APP_ROOT, "index.html")
    
    def extract_text_from_pdf(pdf_path):
        """Extract text from PDF file."""
        try:
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    logger.warning("No PDF library available (PyPDF2 or pdfplumber). Install with: pip install PyPDF2")
                    return None
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None

    def analyze_report_content(text):
        """Analyze extracted text to find tumor types, confidence levels, and generate alerts."""
        if not text:
            return None
        
        text_lower = text.lower()
        analysis = {
            "tumor_types": [],
            "confidence_levels": {},
            "risk_level": "unknown",
            "alerts": [],
            "findings": []
        }
        
        # Detect tumor types
        tumor_keywords = {
            "glioma": ["glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "ependymoma"],
            "meningioma": ["meningioma", "meningeal"],
            "pituitary": ["pituitary", "adenoma", "prolactinoma"],
            "no_tumor": ["no tumor", "normal", "no abnormality", "no mass"]
        }
        
        for tumor_type, keywords in tumor_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if tumor_type not in analysis["tumor_types"]:
                        analysis["tumor_types"].append(tumor_type)
                    break
        
        # Extract confidence levels/percentages
        # Look for patterns like "confidence: 85%", "85% confidence", "85%", etc.
        confidence_patterns = [
            r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*confidence',
            r'(\d+(?:\.\d+)?)\s*%',
            r'probability[:\s]+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*probability'
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    conf_value = float(match)
                    if 0 <= conf_value <= 100:
                        # Find which tumor type this confidence relates to
                        # Look for context around the confidence value
                        for tumor_type in analysis["tumor_types"]:
                            if tumor_type not in analysis["confidence_levels"]:
                                analysis["confidence_levels"][tumor_type] = []
                            analysis["confidence_levels"][tumor_type].append(conf_value)
                except ValueError:
                    pass
        
        # If no specific confidence found, look for general percentages near tumor mentions
        if not analysis["confidence_levels"] and analysis["tumor_types"]:
            # Look for percentages in the text
            percentage_pattern = r'\b(\d+(?:\.\d+)?)\s*%'
            percentages = re.findall(percentage_pattern, text)
            for pct in percentages[:3]:  # Take first few percentages
                try:
                    conf_value = float(pct)
                    if 50 <= conf_value <= 100:  # Likely a confidence level
                        for tumor_type in analysis["tumor_types"]:
                            if tumor_type not in analysis["confidence_levels"]:
                                analysis["confidence_levels"][tumor_type] = []
                            analysis["confidence_levels"][tumor_type].append(conf_value)
                except ValueError:
                    pass
        
        # Determine risk level based on findings
        if "no_tumor" in analysis["tumor_types"]:
            analysis["risk_level"] = "low"
            analysis["alerts"].append("✅ No tumor detected - Low risk")
        elif analysis["tumor_types"]:
            # Check confidence levels
            max_confidence = 0
            for confs in analysis["confidence_levels"].values():
                if confs:
                    max_confidence = max(max_confidence, max(confs))
            
            if max_confidence >= 80:
                analysis["risk_level"] = "high"
                analysis["alerts"].append(f"⚠️ HIGH CONFIDENCE DETECTION ({max_confidence:.1f}%) - Immediate medical consultation recommended")
            elif max_confidence >= 60:
                analysis["risk_level"] = "medium"
                analysis["alerts"].append(f"⚠️ MODERATE CONFIDENCE DETECTION ({max_confidence:.1f}%) - Medical evaluation recommended")
            elif max_confidence > 0:
                analysis["risk_level"] = "low"
                analysis["alerts"].append(f"ℹ️ LOW CONFIDENCE DETECTION ({max_confidence:.1f}%) - Further evaluation may be needed")
            else:
                analysis["risk_level"] = "medium"
                analysis["alerts"].append("⚠️ Tumor detected - Medical consultation recommended")
        
        # Extract key findings
        finding_keywords = ["grade", "size", "location", "enhancement", "edema", "mass effect", "biopsy", "resection"]
        for keyword in finding_keywords:
            if keyword in text_lower:
                analysis["findings"].append(keyword)
        
        # Generate detailed explanation and treatment suggestions
        analysis["detailed_explanation"] = generate_detailed_explanation(analysis)
        analysis["treatment_suggestions"] = generate_treatment_suggestions(analysis)
        analysis["urgency_recommendation"] = generate_urgency_recommendation(analysis)
        
        return analysis

    def generate_detailed_explanation(analysis):
        """Generate a detailed explanation of the report findings."""
        tumor_types = analysis.get("tumor_types", [])
        confidence_levels = analysis.get("confidence_levels", {})
        risk_level = analysis.get("risk_level", "unknown")
        findings = analysis.get("findings", [])
        
        explanation = ""
        
        if "no_tumor" in tumor_types:
            explanation = "**Detailed Analysis:**\n\n"
            explanation += "✅ **No Tumor Detected**\n\n"
            explanation += "Your report indicates that no brain tumor was detected in the imaging study. This is a positive finding.\n\n"
            explanation += "**What this means:**\n"
            explanation += "• The brain imaging appears normal\n"
            explanation += "• No abnormal masses or growths were identified\n"
            explanation += "• No immediate intervention is required\n\n"
            explanation += "**Important Notes:**\n"
            explanation += "• Continue to monitor for any new or worsening symptoms\n"
            explanation += "• Follow your doctor's recommendations for routine follow-up\n"
            explanation += "• If symptoms persist, discuss with your healthcare provider about further evaluation"
        
        elif tumor_types:
            explanation = "**Detailed Analysis:**\n\n"
        
        # Explain detected tumor types
        for tumor_type in tumor_types:
            if tumor_type == "glioma":
                explanation += "**Glioma Detected:**\n"
                explanation += "Gliomas are primary brain tumors that arise from glial cells (supporting cells in the brain).\n\n"
                explanation += "**Key Characteristics:**\n"
                explanation += "• Can be low-grade (slow-growing) or high-grade (aggressive)\n"
                explanation += "• May cause seizures, headaches, and neurological symptoms\n"
                explanation += "• Treatment typically involves surgery, radiation, and/or chemotherapy\n"
                explanation += "• Prognosis depends on grade, location, and molecular markers\n\n"
            
            elif tumor_type == "meningioma":
                explanation += "**Meningioma Detected:**\n"
                explanation += "Meningiomas are tumors that arise from the meninges (protective layers around the brain).\n\n"
                explanation += "**Key Characteristics:**\n"
                explanation += "• Most meningiomas are benign (non-cancerous)\n"
                explanation += "• Usually slow-growing\n"
                explanation += "• Treatment may involve observation, surgery, or radiation\n"
                explanation += "• Generally good prognosis for most cases\n\n"
            
            elif tumor_type == "pituitary":
                explanation += "**Pituitary Tumor Detected:**\n"
                explanation += "Pituitary tumors (adenomas) are growths in the pituitary gland.\n\n"
                explanation += "**Key Characteristics:**\n"
                explanation += "• Most are benign (non-cancerous)\n"
                explanation += "• Can be hormone-secreting or non-functioning\n"
                explanation += "• May cause hormonal imbalances or vision problems\n"
                explanation += "• Treatment involves surgery, medication, or radiation\n\n"
        
        # Add confidence level explanation
        if confidence_levels:
            explanation += "**Confidence Levels:**\n"
            for tumor_type, confs in confidence_levels.items():
                if confs:
                    avg_conf = sum(confs) / len(confs)
                    explanation += f"• {tumor_type.title()}: {avg_conf:.1f}% confidence\n"
                    if avg_conf >= 80:
                        explanation += "  → High confidence indicates strong evidence of this finding\n"
                    elif avg_conf >= 60:
                        explanation += "  → Moderate confidence suggests likely presence, further confirmation may be needed\n"
                    else:
                        explanation += "  → Lower confidence - additional evaluation recommended\n"
            explanation += "\n"
        
        # Add findings explanation
        if findings:
            explanation += "**Key Findings Mentioned in Report:**\n"
            for finding in findings:
                if finding == "grade":
                    explanation += "• **Grade**: Indicates tumor aggressiveness (WHO Grade I-IV)\n"
                elif finding == "size":
                    explanation += "• **Size**: Tumor dimensions and volume\n"
                elif finding == "location":
                    explanation += "• **Location**: Where the tumor is located in the brain\n"
                elif finding == "enhancement":
                    explanation += "• **Enhancement**: How the tumor appears with contrast (may indicate active growth)\n"
                elif finding == "edema":
                    explanation += "• **Edema**: Swelling around the tumor\n"
                elif finding == "mass effect":
                    explanation += "• **Mass Effect**: Compression of surrounding brain structures\n"
            explanation += "\n"
        
        explanation += "**Risk Assessment:**\n"
        if risk_level == "high":
            explanation += "🔴 **HIGH RISK** - The report shows high confidence detection of brain tumor findings. This requires immediate medical attention.\n"
        elif risk_level == "medium":
            explanation += "🟡 **MEDIUM RISK** - The report indicates moderate confidence in tumor detection. Medical evaluation is recommended.\n"
        elif risk_level == "low":
            explanation += "🟢 **LOW RISK** - The report shows low confidence or early-stage findings. Further evaluation may be needed.\n"
        
        return explanation

    def generate_treatment_suggestions(analysis):
        """Generate treatment suggestions based on analysis."""
        tumor_types = analysis.get("tumor_types", [])
        confidence_levels = analysis.get("confidence_levels", {})
        risk_level = analysis.get("risk_level", "unknown")
        
        suggestions = ""
        
        if "no_tumor" in tumor_types:
            suggestions = "**Treatment Recommendations:**\n\n"
            suggestions += "✅ **No Treatment Required**\n\n"
            suggestions += "Since no tumor was detected:\n"
            suggestions += "• No immediate treatment is necessary\n"
            suggestions += "• Continue routine health monitoring\n"
            suggestions += "• Follow your doctor's recommendations for follow-up imaging if symptoms persist\n"
            suggestions += "• Maintain regular health check-ups\n"
        
        elif tumor_types:
            suggestions = "**Treatment Recommendations:**\n\n"
            
            for tumor_type in tumor_types:
                if tumor_type == "glioma":
                    suggestions += "**For Glioma:**\n\n"
                    suggestions += "**Primary Treatment Options:**\n"
                    suggestions += "1. **Surgery** - Maximal safe surgical resection to remove as much tumor as possible\n"
                    suggestions += "2. **Biopsy** - For histologic and molecular profiling (IDH mutation, 1p/19q codeletion)\n"
                    suggestions += "3. **Radiation Therapy** - Often used after surgery, especially for high-grade gliomas\n"
                    suggestions += "4. **Chemotherapy** - Temozolomide is commonly used, especially for glioblastoma\n"
                    suggestions += "5. **Targeted Therapy** - Based on molecular markers (e.g., IDH inhibitors)\n"
                    suggestions += "6. **Supportive Care** - Managing seizures, swelling, and other symptoms\n\n"
                    suggestions += "**Treatment Decision Factors:**\n"
                    suggestions += "• Tumor grade (low-grade vs high-grade)\n"
                    suggestions += "• Molecular markers (IDH status, 1p/19q)\n"
                    suggestions += "• Location and size\n"
                    suggestions += "• Patient's age and overall health\n\n"
                
                elif tumor_type == "meningioma":
                    suggestions += "**For Meningioma:**\n\n"
                    suggestions += "**Primary Treatment Options:**\n"
                    suggestions += "1. **Observation** - Small, asymptomatic meningiomas may be monitored with regular MRIs\n"
                    suggestions += "2. **Surgery** - Surgical resection when feasible, especially for symptomatic or growing tumors\n"
                    suggestions += "3. **Stereotactic Radiosurgery** - For small tumors, skull-base locations, or residual tumors\n"
                    suggestions += "4. **Fractionated Radiation** - For larger tumors or those in critical locations\n"
                    suggestions += "5. **Hormonal Therapy** - Some meningiomas may respond to hormonal treatments\n\n"
                    suggestions += "**Follow-up:**\n"
                    suggestions += "• Postoperative MRI at 3-6 months\n"
                    suggestions += "• Then every 6-12 months depending on WHO grade and symptoms\n"
                    suggestions += "• Most meningiomas are benign (WHO grade I) with good prognosis\n\n"
                
                elif tumor_type == "pituitary":
                    suggestions += "**For Pituitary Tumor:**\n\n"
                    suggestions += "**Primary Treatment Options:**\n"
                    suggestions += "1. **Surgery** - Transsphenoidal resection (through the nose) for symptomatic or growing tumors\n"
                    suggestions += "2. **Medical Therapy** - Dopamine agonists (e.g., cabergoline) for prolactinomas\n"
                    suggestions += "3. **Hormone Replacement** - If pituitary function is compromised\n"
                    suggestions += "4. **Radiation Therapy** - For residual or recurrent tumors\n"
                    suggestions += "5. **Endocrine Management** - Regular hormone level monitoring\n\n"
                    suggestions += "**Follow-up:**\n"
                    suggestions += "• MRI and hormone levels every 3-6 months initially\n"
                    suggestions += "• Long-term monitoring for recurrence\n"
                    suggestions += "• Endocrine evaluation and management\n\n"
        
            suggestions += "**Important:**\n"
            suggestions += "• Treatment should be determined by a multidisciplinary team (neurosurgeon, neuro-oncologist, radiation oncologist)\n"
            suggestions += "• Individual treatment plans are tailored based on specific tumor characteristics\n"
            suggestions += "• Regular MRI surveillance is essential for monitoring treatment response\n"
        
        return suggestions

    def generate_urgency_recommendation(analysis):
        """Generate urgency recommendation (immediate vs scheduled doctor visit)."""
        tumor_types = analysis.get("tumor_types", [])
        confidence_levels = analysis.get("confidence_levels", {})
        risk_level = analysis.get("risk_level", "unknown")
        
        recommendation = {}
        
        if "no_tumor" in tumor_types:
            recommendation["urgency"] = "low"
            recommendation["action"] = "scheduled"
            recommendation["timeframe"] = "Schedule a routine follow-up appointment within 1-3 months"
            recommendation["message"] = "✅ **No Immediate Action Required**\n\n"
            recommendation["message"] += "Since no tumor was detected, you can schedule a routine follow-up appointment with your healthcare provider. No emergency visit is necessary."
        
        elif tumor_types:
            max_confidence = 0
            for confs in confidence_levels.values():
                if confs:
                    max_confidence = max(max_confidence, max(confs))
            
            if risk_level == "high" or max_confidence >= 80:
                recommendation["urgency"] = "high"
                recommendation["action"] = "immediate"
                recommendation["timeframe"] = "Seek medical attention within 24-48 hours or visit emergency department if severe symptoms"
                recommendation["message"] = "🚨 **IMMEDIATE MEDICAL ATTENTION REQUIRED** 🚨\n\n"
                recommendation["message"] += "**Action Required:**\n"
                recommendation["message"] += "• Contact your healthcare provider immediately (within 24-48 hours)\n"
                recommendation["message"] += "• If experiencing severe symptoms (severe headache, seizures, vision loss, confusion), go to the emergency department\n"
                recommendation["message"] += "• Schedule urgent appointment with a neuro-oncologist or neurosurgeon\n"
                recommendation["message"] += "• Do not delay - early intervention is crucial\n\n"
                recommendation["message"] += "**Why Immediate Action:**\n"
                recommendation["message"] += "• High confidence detection indicates strong evidence of brain tumor\n"
                recommendation["message"] += "• Prompt evaluation allows for timely treatment planning\n"
                recommendation["message"] += "• Early intervention may improve treatment outcomes"
            
            elif risk_level == "medium" or (max_confidence >= 60 and max_confidence < 80):
                recommendation["urgency"] = "medium"
                recommendation["action"] = "scheduled_urgent"
                recommendation["timeframe"] = "Schedule appointment within 1-2 weeks"
                recommendation["message"] = "⚠️ **URGENT MEDICAL EVALUATION RECOMMENDED** ⚠️\n\n"
                recommendation["message"] += "**Action Required:**\n"
                recommendation["message"] += "• Schedule an appointment with your healthcare provider within 1-2 weeks\n"
                recommendation["message"] += "• Request referral to a neuro-oncologist or neurosurgeon\n"
                recommendation["message"] += "• If symptoms worsen, seek immediate medical attention\n\n"
                recommendation["message"] += "**Why Urgent Evaluation:**\n"
                recommendation["message"] += "• Moderate confidence detection suggests likely presence of findings\n"
                recommendation["message"] += "• Timely evaluation allows for proper diagnosis and treatment planning\n"
                recommendation["message"] += "• Early assessment can help determine appropriate next steps"
            
            else:
                recommendation["urgency"] = "low"
                recommendation["action"] = "scheduled"
                recommendation["timeframe"] = "Schedule appointment within 2-4 weeks"
                recommendation["message"] = "ℹ️ **SCHEDULED MEDICAL EVALUATION RECOMMENDED** ℹ️\n\n"
                recommendation["message"] += "**Action Required:**\n"
                recommendation["message"] += "• Schedule a routine appointment with your healthcare provider within 2-4 weeks\n"
                recommendation["message"] += "• Discuss the findings and determine if further evaluation is needed\n"
                recommendation["message"] += "• Monitor for any new or worsening symptoms\n\n"
                recommendation["message"] += "**Why Scheduled Evaluation:**\n"
                recommendation["message"] += "• Lower confidence or early findings may require confirmation\n"
                recommendation["message"] += "• Routine evaluation allows for proper assessment\n"
                recommendation["message"] += "• Follow-up imaging may be recommended"
        
        return recommendation

    @app.route("/api/upload-report", methods=["POST"])
    def upload_report():
        if "report" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["report"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Generate unique report ID
        report_id = str(uuid.uuid4())[:8]
        
        # Save file to reports directory
        reports_dir = os.path.join(APP_ROOT, "static", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{report_id}_{file.filename}"
        filepath = os.path.join(reports_dir, filename)
        file.save(filepath)
        
        # Extract and analyze report content
        analysis = None
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
            if text:
                analysis = analyze_report_content(text)
                logger.info(f"Report analyzed: {filename} - Found: {analysis.get('tumor_types', [])}, Risk: {analysis.get('risk_level', 'unknown')}")
        
        # Store report metadata with analysis
        uploaded_reports[report_id] = {
            "filename": file.filename,
            "path": filepath,
            "upload_time": datetime.utcnow().isoformat(),
            "report_id": report_id,
            "analysis": analysis
        }
        
        logger.info(f"Report uploaded: {filename} (ID: {report_id})")
        
        response_data = {
            "report_id": report_id,
            "filename": file.filename,
            "message": "Report uploaded successfully"
        }
        
        # Include analysis results if available
        if analysis:
            response_data["analysis"] = {
                "tumor_types": analysis.get("tumor_types", []),
                "risk_level": analysis.get("risk_level", "unknown"),
                "alerts": analysis.get("alerts", []),
                "detailed_explanation": analysis.get("detailed_explanation", ""),
                "treatment_suggestions": analysis.get("treatment_suggestions", ""),
                "urgency_recommendation": analysis.get("urgency_recommendation", {})
            }
        
        return jsonify(response_data)
    
    @app.route("/api/chat", methods=["POST"])
    def handle_chat():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            user_message = data.get("message", "").strip()
            if not user_message:
                user_message = ""
            user_message_lower = user_message.lower()
            report_id = data.get("report_id")
            
            # Check for specific tumor types first (glioma, meningioma, pituitary)
            tumor_type = None
            if "glioma" in user_message_lower:
                tumor_type = "glioma"
            elif "meningioma" in user_message_lower:
                tumor_type = "meningioma"
            elif "pituitary" in user_message_lower:
                tumor_type = "pituitary"
            
            response_text = None
            
            # Enhanced response logic with specific tumor type information
            if tumor_type:
                # Glioma-specific responses
                if tumor_type == "glioma":
                    if any(word in user_message_lower for word in ["symptom", "sign", "indication"]):
                        response_text = "**Glioma Symptoms:**\n\nGliomas are primary brain tumors that arise from glial cells. Common symptoms include:\n\n• **Headaches** - Often worse in the morning or when lying down\n• **Seizures** - Particularly common in gliomas\n• **Cognitive changes** - Memory problems, confusion, personality changes\n• **Motor symptoms** - Weakness, numbness, or coordination problems\n• **Speech difficulties** - Difficulty finding words or speaking clearly\n• **Vision problems** - Blurred vision, double vision, or loss of peripheral vision\n• **Nausea and vomiting** - Especially in the morning\n• **Fatigue** - Persistent tiredness\n\nGliomas can be low-grade (slow-growing) or high-grade (aggressive). Symptoms vary based on the tumor's location and grade. Early detection and treatment are crucial."
                    elif any(word in user_message_lower for word in ["treatment", "cure", "therapy", "manage"]):
                        response_text = "**Glioma Treatment:**\n\nTreatment for gliomas typically involves:\n\n• **Surgery** - Maximal safe surgical resection to remove as much tumor as possible\n• **Biopsy** - For histologic and molecular profiling (IDH mutation, 1p/19q codeletion)\n• **Radiation Therapy** - Often used after surgery, especially for high-grade gliomas\n• **Chemotherapy** - Temozolomide is commonly used, especially for glioblastoma\n• **Targeted Therapy** - Based on molecular markers (e.g., IDH inhibitors)\n• **Supportive Care** - Managing seizures, swelling, and other symptoms\n\nTreatment depends on:\n- Tumor grade (low-grade vs high-grade)\n- Molecular markers (IDH status, 1p/19q)\n- Location and size\n- Patient's age and overall health\n\nRegular MRI surveillance is essential for monitoring treatment response and recurrence."
                    elif any(word in user_message_lower for word in ["what is", "define", "explain", "about"]):
                        response_text = "**What is a Glioma?**\n\nA glioma is a type of primary brain tumor that originates from glial cells (supporting cells in the brain). Gliomas account for about 30% of all brain tumors.\n\n**Types of Gliomas:**\n• **Astrocytoma** - Arises from astrocytes\n• **Glioblastoma** - Most aggressive type (grade IV)\n• **Oligodendroglioma** - Arises from oligodendrocytes\n• **Ependymoma** - Arises from ependymal cells\n\n**Key Characteristics:**\n• Can be low-grade (slow-growing) or high-grade (aggressive)\n• Location varies - can occur anywhere in the brain\n• May cause seizures, headaches, and neurological deficits\n• Treatment typically involves surgery, radiation, and/or chemotherapy\n• Prognosis depends on grade, location, and molecular markers\n\nGliomas are classified by the World Health Organization (WHO) grading system from grade I (least aggressive) to grade IV (most aggressive)."
                    else:
                        response_text = "**About Gliomas:**\n\nGliomas are primary brain tumors arising from glial cells. They can be low-grade (slow-growing) or high-grade (aggressive like glioblastoma).\n\n**Common characteristics:**\n• Often cause seizures and headaches\n• Treatment involves surgery, radiation, and/or chemotherapy\n• Molecular markers (IDH, 1p/19q) help guide treatment\n• Regular MRI monitoring is essential\n\nWould you like to know more about:\n• Glioma symptoms\n• Glioma treatment options\n• Types of gliomas\n• Prognosis and recovery"
                
                # Meningioma-specific responses
                elif tumor_type == "meningioma":
                    if any(word in user_message_lower for word in ["symptom", "sign", "indication"]):
                        response_text = "**Meningioma Symptoms:**\n\nMeningiomas are tumors that arise from the meninges (protective layers around the brain). Symptoms depend on location and size:\n\n• **Headaches** - Often persistent and gradually worsening\n• **Seizures** - Can occur if the tumor irritates brain tissue\n• **Vision problems** - Blurred vision, double vision, or visual field defects\n• **Hearing loss** - If located near the auditory nerve\n• **Memory and cognitive changes** - Especially with frontal lobe meningiomas\n• **Weakness or numbness** - In arms or legs\n• **Balance problems** - Difficulty walking or coordination\n• **Personality changes** - Less common but possible\n\nMost meningiomas are benign (non-cancerous) and slow-growing. Many are discovered incidentally and may not cause symptoms."
                    elif any(word in user_message_lower for word in ["treatment", "cure", "therapy", "manage"]):
                        response_text = "**Meningioma Treatment:**\n\nTreatment for meningiomas depends on size, location, symptoms, and growth rate:\n\n• **Observation** - Small, asymptomatic meningiomas may be monitored with regular MRIs\n• **Surgery** - Surgical resection when feasible, especially for symptomatic or growing tumors\n• **Stereotactic Radiosurgery** - For small tumors, skull-base locations, or residual tumors after surgery\n• **Fractionated Radiation** - For larger tumors or those in critical locations\n• **Hormonal Therapy** - Some meningiomas may respond to hormonal treatments\n\n**Follow-up:**\n• Postoperative MRI at 3-6 months\n• Then every 6-12 months depending on WHO grade and symptoms\n• Most meningiomas are benign (WHO grade I) with good prognosis\n\nTreatment is individualized based on the tumor's characteristics and patient factors."
                    elif any(word in user_message_lower for word in ["what is", "define", "explain", "about"]):
                        response_text = "**What is a Meningioma?**\n\nA meningioma is a tumor that arises from the meninges - the protective membranes that surround the brain and spinal cord.\n\n**Key Facts:**\n• Most meningiomas are **benign** (non-cancerous)\n• They are the most common primary brain tumor in adults\n• More common in women than men\n• Usually slow-growing\n• Can occur at any age but more common in older adults\n\n**WHO Grading:**\n• **Grade I** - Benign, slow-growing (most common)\n• **Grade II** - Atypical, faster-growing\n• **Grade III** - Anaplastic/malignant, aggressive (rare)\n\n**Common Locations:**\n• Along the surface of the brain\n• Near the skull base\n• Along the spinal cord\n\nMany meningiomas are discovered incidentally and may not require immediate treatment."
                    else:
                        response_text = "**About Meningiomas:**\n\nMeningiomas are tumors arising from the meninges (protective brain coverings). Most are benign and slow-growing.\n\n**Key points:**\n• Usually non-cancerous (benign)\n• Treatment may involve observation, surgery, or radiation\n• Good prognosis for most cases\n• Regular monitoring with MRI\n\nWould you like to know more about:\n• Meningioma symptoms\n• Meningioma treatment options\n• Prognosis and recovery"
                
                # Pituitary-specific responses
                elif tumor_type == "pituitary":
                    if any(word in user_message_lower for word in ["symptom", "sign", "indication"]):
                        response_text = "**Pituitary Tumor Symptoms:**\n\nPituitary tumors (adenomas) can cause symptoms related to:\n\n**Hormonal Effects:**\n• **Prolactinoma** - Irregular periods, infertility, milk production, low libido\n• **Growth hormone** - Acromegaly (enlarged hands/feet), joint pain\n• **ACTH** - Cushing's syndrome (weight gain, high blood pressure)\n• **Thyroid-stimulating** - Hyperthyroidism symptoms\n\n**Mass Effects (from tumor size):**\n• **Vision problems** - Loss of peripheral vision (bitemporal hemianopsia)\n• **Headaches** - Often persistent\n• **Double vision** - If tumor compresses nerves\n• **Nausea and vomiting**\n• **Fatigue and weakness**\n\n**Hormone Deficiency Symptoms:**\n• Fatigue, weakness\n• Low blood pressure\n• Weight loss or gain\n• Sexual dysfunction\n\nSymptoms vary greatly depending on whether the tumor is hormone-secreting and its size."
                    elif any(word in user_message_lower for word in ["treatment", "cure", "therapy", "manage"]):
                        response_text = "**Pituitary Tumor Treatment:**\n\nTreatment depends on tumor type, size, and hormone production:\n\n**Surgery:**\n• **Transsphenoidal resection** - Most common approach through the nose\n• Minimally invasive, preserves pituitary function when possible\n• Used for symptomatic or growing tumors\n\n**Medical Therapy:**\n• **Dopamine agonists** (e.g., cabergoline) - For prolactinomas\n• **Somatostatin analogs** - For growth hormone-secreting tumors\n• **Hormone replacement** - If pituitary function is compromised\n\n**Radiation Therapy:**\n• Used for residual or recurrent tumors\n• Stereotactic radiosurgery for small residual tumors\n• Fractionated radiation for larger tumors\n\n**Follow-up:**\n• MRI and hormone levels every 3-6 months initially\n• Long-term monitoring for recurrence\n• Endocrine evaluation and management\n\nTreatment is highly individualized based on tumor characteristics."
                    elif any(word in user_message_lower for word in ["what is", "define", "explain", "about"]):
                        response_text = "**What is a Pituitary Tumor?**\n\nA pituitary tumor (adenoma) is a growth in the pituitary gland - a small gland at the base of the brain that controls hormone production.\n\n**Key Facts:**\n• Most pituitary tumors are **benign** (non-cancerous)\n• Can be **functioning** (hormone-secreting) or **non-functioning**\n• Located at the base of the brain, near the optic nerves\n• Can affect vision if large enough\n\n**Types:**\n• **Prolactinoma** - Most common, secretes prolactin\n• **Growth hormone-secreting** - Causes acromegaly\n• **ACTH-secreting** - Causes Cushing's disease\n• **Non-functioning** - No hormone production\n\n**Common Issues:**\n• Hormone imbalances\n• Vision problems (if large)\n• Headaches\n• Pituitary function impairment\n\nMany pituitary tumors are successfully treated with surgery and/or medication."
                    else:
                        response_text = "**About Pituitary Tumors:**\n\nPituitary tumors (adenomas) are growths in the pituitary gland that controls hormones.\n\n**Key points:**\n• Usually benign\n• Can be hormone-secreting or non-functioning\n• Treatment: surgery, medication, or radiation\n• Good prognosis with proper treatment\n\nWould you like to know more about:\n• Pituitary tumor symptoms\n• Pituitary tumor treatment\n• Types of pituitary tumors"
            
            # Generic responses (if no specific tumor type or specific question not matched)
            if not response_text:
                responses = {
                    "what is a brain tumor": "A brain tumor is a mass or growth of abnormal cells in the brain. Tumors can be benign (non-cancerous) or malignant (cancerous). Primary brain tumors originate in the brain, while metastatic brain tumors spread from other parts of the body.",
                    "symptom": "Common symptoms of brain tumors may include:\n• Persistent headaches that worsen over time\n• Seizures or convulsions\n• Nausea or vomiting\n• Vision or hearing problems\n• Balance difficulties\n• Personality or behavior changes\n• Memory problems\n• Speech difficulties\n\nIf you experience these symptoms, please consult a healthcare professional for proper evaluation.",
                    "treatment": "Treatment for brain tumors depends on the type, size, location, and grade of the tumor, as well as the patient's overall health. Common treatment options include:\n\n• Surgery: To remove as much of the tumor as possible\n• Radiation Therapy: Using high-energy beams to destroy tumor cells\n• Chemotherapy: Using drugs to kill cancer cells\n• Targeted Drug Therapy: Focusing on specific abnormalities in cancer cells\n• Immunotherapy: Helping the immune system fight cancer\n\nTreatment is often a combination of these approaches, determined by a team of specialists.",
                    "recovery": "Recovery from brain tumor treatment depends on several factors:\n• Tumor type and grade\n• Size and location\n• Extent of surgical resection\n• Patient's age and overall health\n• Response to treatment\n\nRehabilitation may include physical therapy, occupational therapy, and speech therapy. Follow-up care is essential for monitoring and managing any complications.",
                    "cause": "The exact cause of most brain tumors is unknown. However, certain risk factors may increase the likelihood:\n• Genetic factors and family history\n• Exposure to ionizing radiation\n• Age (more common in older adults and children)\n• Previous cancer history\n• Certain genetic conditions\n\nMost brain tumors develop without a known cause.",
                    "type": "Main types of brain tumors include:\n\n1. **Gliomas**: Originate from glial cells (astrocytoma, glioblastoma, oligodendroglioma)\n2. **Meningiomas**: Arise from the meninges (usually benign)\n3. **Pituitary Adenomas**: Develop in the pituitary gland\n4. Medulloblastomas: Common in children\n5. Schwannomas: Develop from nerve sheath cells\n6. Metastatic Brain Tumors: Spread from other cancers in the body",
                    "diagnos": "Diagnosis of brain tumors typically involves:\n\n• Neurological Examination: Testing vision, hearing, balance, coordination, and reflexes\n• Imaging Tests:\n  - MRI (Magnetic Resonance Imaging)\n  - CT (Computed Tomography) scan\n  - PET scan\n• Biopsy: Removing a sample of tissue for analysis\n• Additional Tests: Blood tests, lumbar puncture, or specialized imaging\n\nEarly and accurate diagnosis is crucial for effective treatment planning.",
                    "hello": "Hello! I'm your AI medical assistant. I can help you with information about brain tumors, including symptoms, diagnosis, treatment options, and general medical information. How can I assist you today?",
                    "hi": "Hi there! I'm here to help answer your questions about brain tumors and related medical information. What would you like to know?",
                    "help": "I can help you with information about:\n\n• Brain tumor symptoms and signs\n• Diagnosis methods\n• Treatment options (surgery, radiation, chemotherapy)\n• Types of brain tumors (glioma, meningioma, pituitary)\n• Recovery and prognosis\n• General brain tumor information\n\nPlease ask me any specific questions you have, and I'll do my best to provide helpful information.",
                    "mri": "MRI (Magnetic Resonance Imaging) is a key diagnostic tool for brain tumors. It provides detailed images of the brain without using radiation.\n\n**What MRI shows:**\n• Tumor location, size, and shape\n• Relationship to surrounding brain structures\n• Presence of swelling (edema)\n• Blood flow patterns\n• Response to treatment\n\n**Types of MRI scans:**\n• T1-weighted: Shows anatomy\n• T2-weighted: Shows fluid and edema\n• Contrast-enhanced: Highlights areas with disrupted blood-brain barrier\n• Diffusion-weighted: Shows areas of restricted diffusion\n\nMRI results are interpreted by radiologists and neuro-oncologists to guide diagnosis and treatment planning.",
                    "result": "I can help you understand MRI results. Common findings in brain tumor MRI scans include:\n\n• **Abnormal masses** - Areas that don't match normal brain tissue\n• **Enhancement** - Areas that light up with contrast (may indicate active tumor)\n• **Edema** - Swelling around the tumor\n• **Mass effect** - Compression or shifting of brain structures\n• **Hemorrhage** - Bleeding within or around the tumor\n\n**Important:** MRI results should always be interpreted by qualified medical professionals. I can provide general information, but cannot diagnose or replace professional medical advice.\n\nWould you like to know more about:\n• Specific tumor types (glioma, meningioma, pituitary)\n• Treatment options\n• What to expect after diagnosis",
                    "explain": "I can help explain brain tumor-related information. Please be specific about what you'd like me to explain:\n\n• **MRI results** - I can explain common findings and terminology\n• **Tumor types** - Information about glioma, meningioma, pituitary tumors\n• **Symptoms** - What to watch for\n• **Treatment options** - Surgery, radiation, chemotherapy\n• **Medical terms** - Definitions and explanations\n\nFor example, you could ask:\n• 'Explain glioma symptoms'\n• 'What does enhancement mean on an MRI?'\n• 'Explain the treatment options for meningioma'\n\nWhat would you like me to explain?",
                }
                
                # Find the best matching response
                for key, value in responses.items():
                    if key in user_message_lower:
                        response_text = value
                        break
            
            # If report is uploaded, provide context-aware responses
            if report_id and report_id in uploaded_reports:
                report_info = uploaded_reports[report_id]
                report_filename = report_info['filename']
                report_analysis = report_info.get('analysis')
                
                # Generate alert message if analysis is available
                alert_message = ""
                if report_analysis:
                    alerts = report_analysis.get('alerts', [])
                    tumor_types = report_analysis.get('tumor_types', [])
                    confidence_levels = report_analysis.get('confidence_levels', {})
                    risk_level = report_analysis.get('risk_level', 'unknown')
                    
                    if alerts:
                        alert_message = "\n\n🚨 REPORT ANALYSIS ALERTS 🚨\n\n"
                        
                        for alert in alerts:
                            alert_message += f"{alert}\n"
                        
                        if tumor_types:
                            alert_message += f"\nDetected Tumor Types: {', '.join(tumor_types).title()}\n"
                        
                        if confidence_levels:
                            alert_message += "\nConfidence Levels:\n"
                            for tumor_type, confs in confidence_levels.items():
                                if confs:
                                    avg_conf = sum(confs) / len(confs)
                                    alert_message += f"• {tumor_type.title()}: {avg_conf:.1f}%\n"
                        
                        alert_message += f"\nOverall Risk Level: {risk_level.upper()}\n"
                
                # Check if user is asking a specific question about the report
                if user_message_lower:
                    # If user asked a specific question, answer it in context of the report
                    if any(word in user_message_lower for word in ["risk", "risk factor", "danger", "concern", "problem", "issue", "alert"]):
                        response_text = f"Regarding your uploaded report '{report_filename}':\n\n"
                        
                        if report_analysis:
                            alerts = report_analysis.get('alerts', [])
                            risk_level = report_analysis.get('risk_level', 'unknown')
                            tumor_types = report_analysis.get('tumor_types', [])
                            confidence_levels = report_analysis.get('confidence_levels', {})
                            
                            response_text += alert_message
                            
                            response_text += "\nRisk Assessment:\n"
                            if risk_level == "high":
                                response_text += "🔴 HIGH RISK - Immediate medical consultation required. High confidence detection found.\n\n"
                            elif risk_level == "medium":
                                response_text += "🟡 MEDIUM RISK - Medical evaluation recommended. Moderate confidence in tumor detection.\n\n"
                            elif risk_level == "low":
                                response_text += "🟢 LOW RISK - Low confidence or no tumor detected. Follow-up with healthcare provider advised.\n\n"
                            
                            if confidence_levels:
                                response_text += "Confidence Analysis:\n"
                                for tumor_type, confs in confidence_levels.items():
                                    if confs:
                                        avg_conf = sum(confs) / len(confs)
                                        if avg_conf >= 80:
                                            response_text += f"• {tumor_type.title()}: {avg_conf:.1f}% (HIGH CONFIDENCE) ⚠️\n"
                                        elif avg_conf >= 60:
                                            response_text += f"• {tumor_type.title()}: {avg_conf:.1f}% (MODERATE CONFIDENCE) ⚠️\n"
                                        else:
                                            response_text += f"• {tumor_type.title()}: {avg_conf:.1f}% (LOW CONFIDENCE)\n"
                                response_text += "\n"
                        else:
                            response_text += "\nBased on brain tumor risk factors, here are important considerations:\n\n**General Risk Factors:**\n• **Age** - Brain tumors can occur at any age, but some types are more common in certain age groups\n• **Gender** - Some tumor types (like meningiomas) are more common in women\n• **Family History** - Genetic factors may play a role in some cases\n• **Exposure to Radiation** - Previous radiation therapy increases risk\n• **Previous Cancer** - History of cancer may increase risk of metastatic brain tumors\n\n**Tumor-Specific Risk Factors:**\n• **Glioma** - Genetic mutations (IDH, 1p/19q), age, radiation exposure\n• **Meningioma** - Female gender, age, hormone factors\n• **Pituitary** - Genetic syndromes, family history\n\n**Important:** Please consult with your healthcare provider for a detailed analysis of your specific case and any risk factors identified in your report."
                    
                    elif any(word in user_message_lower for word in ["analyze", "analysis", "review", "examine"]):
                        response_text = f"**Analysis of your uploaded report '{report_filename}':**\n\nI can help you understand your report findings. Based on brain tumor analysis reports, here's what to look for:\n\n**Key Findings Typically Include:**\n• **Tumor Type** - Classification (glioma, meningioma, pituitary, etc.)\n• **Location** - Where the tumor is located in the brain\n• **Size** - Dimensions and volume measurements\n• **Grade** - WHO grading (I-IV) indicating aggressiveness\n• **Enhancement Pattern** - How the tumor appears on contrast imaging\n• **Mass Effect** - Impact on surrounding brain structures\n• **Edema** - Swelling around the tumor\n\n**What to Discuss with Your Doctor:**\n• Specific findings in your report\n• Treatment recommendations based on the findings\n• Follow-up imaging schedule\n• Prognosis and expected outcomes\n\nFor a detailed analysis of your specific report, please share the key findings or ask specific questions about the terminology used in your report."
                    
                    elif any(word in user_message_lower for word in ["explain", "what does", "meaning", "understand"]):
                        response_text = f"**Explaining your report '{report_filename}':**\n\nI can help explain medical terms and findings in your report. Common terms in brain tumor reports include:\n\n**Imaging Terms:**\n• **Enhancement** - Areas that 'light up' with contrast, often indicating active tumor\n• **Mass Effect** - Compression or shifting of brain structures\n• **Edema** - Swelling around the tumor\n• **T1/T2 weighted** - Different MRI sequences showing different tissue characteristics\n\n**Tumor Characteristics:**\n• **Grade** - WHO grading system (I=least aggressive, IV=most aggressive)\n• **Resection** - Surgical removal\n• **Biopsy** - Tissue sample for diagnosis\n• **Molecular markers** - Genetic characteristics (IDH, 1p/19q, etc.)\n\n**What would you like me to explain?** Please share specific terms or findings from your report, and I'll provide detailed explanations.\n\nFor example:\n• 'What does enhancement mean?'\n• 'Explain the grade in my report'\n• 'What is mass effect?'"
                    
                    elif any(word in user_message_lower for word in ["result", "findings", "summary", "conclusion"]):
                        response_text = f"**Summary of your report '{report_filename}':**\n\nTo provide a comprehensive summary of your report findings, I would typically look for:\n\n**Report Components:**\n• **Clinical History** - Patient symptoms and background\n• **Imaging Findings** - MRI/CT scan observations\n• **Diagnosis** - Tumor type and classification\n• **Recommendations** - Suggested next steps\n\n**Common Findings in Brain Tumor Reports:**\n• Tumor location and size\n• Enhancement characteristics\n• Mass effect on surrounding structures\n• Edema (swelling) extent\n• Grade and aggressiveness indicators\n\n**Next Steps Typically Include:**\n• Consultation with neuro-oncologist\n• Treatment planning (surgery, radiation, chemotherapy)\n• Follow-up imaging schedule\n• Monitoring plan\n\n**To get a specific summary of your report:** Please share the key findings or diagnosis mentioned in your report, and I can provide more detailed explanations and guidance."
                    
                    elif any(word in user_message_lower for word in ["treatment", "therapy", "cure", "manage", "what to do", "procedure", "procedure to cure"]):
                        response_text = f"**Treatment recommendations for your report '{report_filename}':**\n\n"
                        
                        # If report analysis is available, use it for context-aware treatment
                        if report_analysis:
                            tumor_types = report_analysis.get('tumor_types', [])
                            risk_level = report_analysis.get('risk_level', 'unknown')
                            
                            if report_analysis.get('treatment_suggestions'):
                                response_text += report_analysis.get('treatment_suggestions')
                                response_text += "\n\n"
                            
                            # Add urgency recommendation
                            urgency_rec = report_analysis.get('urgency_recommendation', {})
                            if urgency_rec:
                                response_text += "Next Steps:\n"
                                msg = urgency_rec.get('message', '')
                                # Simplify urgency message
                                if "IMMEDIATE" in msg:
                                    response_text += "IMMEDIATE ACTION REQUIRED\n"
                                    response_text += "• Contact healthcare provider WITHIN 24-48 HOURS\n"
                                    response_text += "• Go to emergency department if severe symptoms\n\n"
                                elif "URGENT" in msg:
                                    response_text += "URGENT ACTION REQUIRED\n"
                                    response_text += "• Schedule appointment WITHIN 1-2 WEEKS\n"
                                    response_text += "• Request referral to specialist\n\n"
                                else:
                                    response_text += "SCHEDULED EVALUATION RECOMMENDED\n"
                                    response_text += "• Schedule appointment within 2-4 weeks\n\n"
                                response_text += f"Timeframe: {urgency_rec.get('timeframe', 'Consult with your doctor')}\n\n"
                        else:
                            response_text += "Treatment options depend on the specific findings in your report. Here are general treatment approaches:\n\n**Surgical Options:**\n• **Biopsy** - To confirm diagnosis and determine tumor type\n• **Resection** - Surgical removal (maximal safe resection)\n• **Minimally invasive** - For certain tumor types and locations\n\n**Non-Surgical Options:**\n• **Radiation Therapy** - External beam or stereotactic radiosurgery\n• **Chemotherapy** - Drug-based treatment\n• **Targeted Therapy** - Based on molecular markers\n• **Observation** - For slow-growing, asymptomatic tumors\n\n**Treatment Decision Factors:**\n• Tumor type and grade\n• Location and size\n• Patient age and overall health\n• Symptoms and functional impact\n\n**Important:** Treatment recommendations should be made by your medical team based on your specific report findings. Please discuss the treatment options with your neuro-oncologist.\n\nWould you like information about treatment for a specific tumor type (glioma, meningioma, pituitary)?"
                    
                    elif any(word in user_message_lower for word in ["precaution", "precautions", "care", "lifestyle", "what should i avoid", "what not to do", "avoid", "preventing", "preventing measures", "prevention", "prevent", "preventive"]):
                        response_text = f"Precautions and Preventing Measures for your report '{report_filename}':\n\n"
                        
                        if report_analysis:
                            tumor_types = report_analysis.get('tumor_types', [])
                            risk_level = report_analysis.get('risk_level', 'unknown')
                            confidence_levels = report_analysis.get('confidence_levels', {})
                            alerts = report_analysis.get('alerts', [])
                            
                            # Show specific report findings first
                            response_text += "📋 Based on Your Report Analysis:\n\n"
                            
                            if tumor_types:
                                detected_tumors = [t for t in tumor_types if t != 'no_tumor']
                                if detected_tumors:
                                    response_text += f"Detected Tumor Type(s): {', '.join([t.title() for t in detected_tumors])}\n"
                                    if confidence_levels:
                                        for tumor_type, confs in confidence_levels.items():
                                            if confs:
                                                avg_conf = sum(confs) / len(confs)
                                                response_text += f"• {tumor_type.title()}: {avg_conf:.1f}% confidence detected\n"
                                    response_text += "\n"
                            
                            if risk_level:
                                response_text += f"Risk Level: {risk_level.upper()}\n\n"
                            
                            if alerts:
                                response_text += "Report Alerts:\n"
                                for alert in alerts:
                                    response_text += f"• {alert}\n"
                                response_text += "\n"
                            
                            response_text += "🚨 SPECIFIC PRECAUTIONS FOR YOUR REPORT 🚨\n\n"
                            
                            if risk_level == "high":
                                response_text += "HIGH PRIORITY PRECAUTIONS\n\n"
                                response_text += "Based on your HIGH RISK detection:\n"
                                response_text += "• URGENT: Seek immediate medical attention if severe symptoms occur\n"
                                response_text += "• URGENT: Schedule appointment WITHIN 24-48 HOURS\n"
                                response_text += "• CRITICAL: Avoid driving until cleared by doctor\n"
                                response_text += "• CRITICAL: Do not take new medications without doctor's approval\n"
                                response_text += "• Keep emergency contact information available\n\n"
                            elif risk_level == "medium":
                                response_text += "MODERATE PRIORITY PRECAUTIONS\n\n"
                                response_text += "Based on your MODERATE RISK detection:\n"
                                response_text += "• IMPORTANT: Schedule medical evaluation WITHIN 1-2 WEEKS\n"
                                response_text += "• MONITOR CLOSELY: Watch for symptoms - headaches, vision changes, seizures\n"
                                response_text += "• SEEK IMMEDIATE HELP if symptoms worsen\n"
                                response_text += "• Avoid activities that could cause head injury\n"
                                response_text += "• Inform doctor about any new or worsening symptoms\n\n"
                            else:
                                response_text += "GENERAL PRECAUTIONS\n\n"
                                response_text += "Based on your LOW RISK detection:\n"
                                response_text += "• Schedule routine follow-up within 2-4 weeks\n"
                                response_text += "• Monitor for any new symptoms\n"
                                response_text += "• Maintain regular health check-ups\n\n"
                            
                            # Tumor-specific precautions
                            if tumor_types:
                                detected_tumors = [t for t in tumor_types if t != 'no_tumor']
                                if detected_tumors:
                                    response_text += "TUMOR-SPECIFIC PRECAUTIONS (Based on Your Report):\n\n"
                                    for tumor_type in detected_tumors:
                                        if tumor_type == "glioma":
                                            response_text += "For Glioma:\n"
                                            response_text += "• Monitor for seizures - avoid triggers\n"
                                            response_text += "• Take prescribed anti-seizure medications as directed\n"
                                            response_text += "• Report neurological changes immediately\n"
                                            response_text += "• Avoid activities that increase intracranial pressure\n\n"
                                        elif tumor_type == "meningioma":
                                            response_text += "For Meningioma:\n"
                                            response_text += "• Monitor for vision changes - report immediately\n"
                                            response_text += "• Track headache patterns\n"
                                            response_text += "• Regular imaging follow-up (every 3-6 months)\n"
                                            response_text += "• Report any new symptoms promptly\n\n"
                                        elif tumor_type == "pituitary":
                                            response_text += "For Pituitary Tumor:\n"
                                            response_text += "• Regular hormone level monitoring\n"
                                            response_text += "• Report vision changes immediately\n"
                                            response_text += "• Take hormone replacement if prescribed\n"
                                            response_text += "• Monitor hormonal symptoms\n\n"
                            
                            response_text += "General Lifestyle Precautions:\n\n"
                            response_text += "• Avoid head injuries - Wear helmets, avoid high-risk activities\n"
                            response_text += "• Manage stress - Practice relaxation, get adequate sleep\n"
                            response_text += "• Healthy diet - Balanced nutrition, stay hydrated\n"
                            response_text += "• Medication safety - Do not take supplements without doctor's approval\n"
                            response_text += "• Avoid alcohol and smoking\n"
                            response_text += "• Consult doctor before starting exercise\n\n"
                        else:
                            response_text += "General Precautions:\n\n"
                            response_text += "• Seek immediate medical attention for severe symptoms\n"
                            response_text += "• Avoid head injuries and high-risk activities\n"
                            response_text += "• Manage stress and maintain healthy lifestyle\n"
                            response_text += "• Do not take medications without doctor's approval\n"
                            response_text += "• Keep emergency contact information available\n\n"
                        
                        response_text += "Important: Always consult with your healthcare provider for precautions specific to your condition."
                    
                    elif any(word in user_message_lower for word in ["home remedy", "home remedies", "natural treatment", "herbal", "diet", "supplement", "alternative"]):
                        response_text = f"**Home Remedies and Supportive Care for your report '{report_filename}':**\n\n"
                        
                        if report_analysis:
                            tumor_types = report_analysis.get('tumor_types', [])
                            risk_level = report_analysis.get('risk_level', 'unknown')
                            
                            response_text += "⚠️ **IMPORTANT NOTE:** Home remedies and natural treatments should **NOT** replace medical treatment. Always consult your doctor before trying any home remedies.\n\n"
                            
                            response_text += "**Supportive Home Care (Complementary to Medical Treatment):**\n\n"
                            
                            response_text += "**1. Nutrition and Diet:**\n"
                            response_text += "• **Anti-inflammatory foods** - Include berries, leafy greens, fatty fish (omega-3), nuts\n"
                            response_text += "• **Antioxidant-rich foods** - Colorful fruits and vegetables, green tea\n"
                            response_text += "• **Hydration** - Drink plenty of water (consult doctor if fluid restriction needed)\n"
                            response_text += "• **Avoid processed foods** - Limit sugar, processed meats, refined foods\n"
                            response_text += "• **Small, frequent meals** - If experiencing nausea or appetite issues\n\n"
                            
                            response_text += "**2. Stress Management:**\n"
                            response_text += "• **Meditation and mindfulness** - Reduce stress, improve mental well-being\n"
                            response_text += "• **Yoga or gentle exercise** - Consult doctor first, especially if balance is affected\n"
                            response_text += "• **Adequate sleep** - 7-9 hours per night, maintain regular sleep schedule\n"
                            response_text += "• **Relaxation techniques** - Deep breathing, progressive muscle relaxation\n\n"
                            
                            response_text += "**3. Symptom Management:**\n"
                            response_text += "• **Headaches** - Rest in dark, quiet room; cool compress; stay hydrated\n"
                            response_text += "• **Nausea** - Ginger tea (consult doctor), small frequent meals, avoid strong odors\n"
                            response_text += "• **Fatigue** - Pace activities, prioritize rest, light exercise if approved\n"
                            response_text += "• **Sleep issues** - Maintain sleep routine, avoid screens before bed, comfortable environment\n\n"
                            
                            response_text += "**4. General Wellness:**\n"
                            response_text += "• **Gentle exercise** - Walking, stretching (only if approved by doctor)\n"
                            response_text += "• **Social support** - Stay connected with family and friends\n"
                            response_text += "• **Stay informed** - Learn about your condition from reliable medical sources\n"
                            response_text += "• **Keep a symptom journal** - Track symptoms, medications, triggers\n\n"
                            
                            response_text += "**⚠️ CAUTIONS - What to AVOID:**\n"
                            response_text += "• **Do NOT** replace medical treatment with home remedies\n"
                            response_text += "• **Do NOT** take supplements without doctor's approval (can interact with medications)\n"
                            response_text += "• **Do NOT** try unproven 'cures' or expensive alternative treatments\n"
                            response_text += "• **Do NOT** ignore worsening symptoms - seek medical attention\n"
                            response_text += "• **Avoid** high-dose vitamins or herbal remedies without medical supervision\n\n"
                            
                            if tumor_types:
                                response_text += "**Tumor-Specific Supportive Care:**\n\n"
                                for tumor_type in tumor_types:
                                    if tumor_type == "glioma":
                                        response_text += "**For Glioma - Additional Support:**\n"
                                        response_text += "• Support cognitive function with mental exercises (if approved)\n"
                                        response_text += "• Monitor for seizure triggers and avoid them\n"
                                        response_text += "• Maintain medication schedule strictly\n"
                                        response_text += "• Support neurological recovery through approved therapies\n\n"
                                    elif tumor_type == "meningioma":
                                        response_text += "**For Meningioma - Additional Support:**\n"
                                        response_text += "• Monitor headaches and vision changes\n"
                                        response_text += "• Support brain health with approved activities\n"
                                        response_text += "• Manage stress which can affect symptoms\n\n"
                                    elif tumor_type == "pituitary":
                                        response_text += "**For Pituitary Tumor - Additional Support:**\n"
                                        response_text += "• Support hormonal balance through diet and lifestyle (as approved)\n"
                                        response_text += "• Monitor energy levels and adjust activities accordingly\n"
                                        response_text += "• Support vision health if affected\n\n"
                        else:
                            response_text += "**General Supportive Home Care:**\n\n"
                            response_text += "• Healthy, balanced diet with anti-inflammatory foods\n"
                            response_text += "• Stress management through meditation, yoga (consult doctor first)\n"
                            response_text += "• Adequate rest and sleep\n"
                            response_text += "• Gentle exercise if approved by doctor\n"
                            response_text += "• Stay hydrated\n"
                            response_text += "• Social support and mental well-being\n\n"
                            response_text += "**⚠️ Important:** Always consult your healthcare provider before trying any home remedies or supplements. They should complement, not replace, medical treatment.\n"
                        
                        response_text += "\n**Remember:** Home remedies are supportive care only. Medical treatment is essential for brain tumor management. Always follow your doctor's treatment plan."
                    
                    elif any(word in user_message_lower for word in ["symptom", "sign", "indication"]):
                        response_text = f"**Symptoms related to your report '{report_filename}':**\n\nSymptoms depend on the tumor type and location identified in your report. Common symptoms include:\n\n**General Symptoms:**\n• Headaches (often worse in morning)\n• Seizures\n• Nausea and vomiting\n• Vision or hearing problems\n• Balance and coordination issues\n• Cognitive changes (memory, personality)\n• Weakness or numbness\n\n**Location-Specific Symptoms:**\n• **Frontal lobe** - Personality changes, cognitive issues\n• **Temporal lobe** - Memory problems, seizures\n• **Parietal lobe** - Sensory changes, spatial awareness\n• **Occipital lobe** - Vision problems\n• **Brainstem** - Balance, coordination, vital functions\n• **Pituitary** - Hormonal changes, vision problems\n\n**What to Monitor:**\n• Any new or worsening symptoms\n• Changes in cognitive function\n• Seizure activity\n• Vision or hearing changes\n\nPlease consult with your healthcare provider about symptoms specific to your diagnosis."
                    
                    # If we have a response_text from earlier matching, enhance it with report context
                    elif response_text:
                        response_text = f"**Regarding your uploaded report '{report_filename}':**\n\n{response_text}\n\n**Note:** This information is general. For specific guidance related to your report findings, please consult with your healthcare provider."
                    
                    # If no specific match but user asked something, provide helpful response
                    else:
                        response_text = f"**Regarding your uploaded report '{report_filename}':**"
                        
                        # Add alerts if analysis is available
                        if report_analysis:
                            response_text += alert_message
                        
                        response_text += "\n\nI can help you understand your report.\n\nI can help with:\n• Explaining medical terms and findings\n• Understanding risk factors and alerts\n• Treatment options\n• Next steps\n\nAsk me specific questions about your report findings."
                else:
                    # Empty message but report uploaded - show complete analysis immediately
                    response_text = f"📋 Report Analysis: {report_filename}\n\n"
                    
                    # Add complete analysis if available
                    if report_analysis:
                        # Show Risk Assessment FIRST (most important) - HIGHLIGHTED
                        risk_level = report_analysis.get('risk_level', 'unknown')
                        tumor_types = report_analysis.get('tumor_types', [])
                        confidence_levels = report_analysis.get('confidence_levels', {})
                        
                        # RISK LEVEL SUMMARY
                        if risk_level == "high":
                            response_text += "🚨 RISK LEVEL: HIGH 🚨\n\n"
                            response_text += "HIGH RISK DETECTED\n\n"
                            response_text += "URGENT ACTION REQUIRED\n"
                            response_text += "• Contact healthcare provider WITHIN 24-48 HOURS\n"
                            response_text += "• If severe symptoms (headache, seizures, vision loss), go to emergency department\n"
                            response_text += "• Schedule urgent appointment with neuro-oncologist\n\n"
                        elif risk_level == "medium":
                            response_text += "⚠️ RISK LEVEL: MEDIUM ⚠️\n\n"
                            response_text += "MODERATE RISK DETECTED\n\n"
                            response_text += "ACTION REQUIRED\n"
                            response_text += "• Schedule appointment WITHIN 1-2 WEEKS\n"
                            response_text += "• Request referral to neuro-oncologist or neurosurgeon\n"
                            response_text += "• Monitor symptoms closely - seek immediate help if they worsen\n\n"
                        elif risk_level == "low":
                            if "no_tumor" in tumor_types:
                                response_text += "✅ RISK LEVEL: LOW ✅\n\n"
                                response_text += "NO TUMOR DETECTED\n\n"
                                response_text += "ACTION:\n"
                                response_text += "• Routine follow-up recommended\n"
                                response_text += "• Continue regular health monitoring\n\n"
                            else:
                                response_text += "ℹ️ RISK LEVEL: LOW ℹ️\n\n"
                                response_text += "LOW RISK DETECTED\n\n"
                                response_text += "ACTION:\n"
                                response_text += "• Schedule appointment within 2-4 weeks\n"
                                response_text += "• Monitor for any new symptoms\n\n"
                        else:
                            response_text += "❓ RISK LEVEL: UNKNOWN ❓\n\n"
                            response_text += "Status: Analysis incomplete\n"
                            response_text += "Action: Please consult with your healthcare provider\n\n"
                        
                        # Confidence Levels
                        if confidence_levels:
                            response_text += "📊 Confidence Levels:\n"
                            for tumor_type, confs in confidence_levels.items():
                                if confs:
                                    avg_conf = sum(confs) / len(confs)
                                    response_text += f"\n{tumor_type.title()}: {avg_conf:.1f}%"
                                    if avg_conf >= 80:
                                        response_text += " (High confidence) ⚠️\n"
                                    elif avg_conf >= 60:
                                        response_text += " (Moderate confidence) ⚠️\n"
                                    else:
                                        response_text += " (Low confidence)\n"
                            response_text += "\n"
                        elif tumor_types:
                            response_text += "Tumor Types Detected:\n"
                            for tumor_type in tumor_types:
                                response_text += f"• {tumor_type.title()}\n"
                            response_text += "\nNote: Confidence levels not found in report\n\n"
                        
                        # RISK FACTORS SECTION
                        response_text += "\n📊 RISK FACTORS ANALYSIS 📊\n\n"
                        
                        if "no_tumor" in tumor_types:
                            response_text += "✅ No brain tumor detected\n\n"
                            response_text += "Risk Assessment:\n"
                            response_text += "• Tumor Risk: None\n"
                            response_text += "• Immediate Risk: Low\n"
                            response_text += "• Action: Routine follow-up\n\n"
                        elif tumor_types:
                            response_text += "⚠️ Brain tumor detected in your report\n\n"
                            
                            for tumor_type in tumor_types:
                                if tumor_type == "glioma":
                                    response_text += "Glioma Risk Factors:\n"
                                    response_text += "• Genetic mutations (IDH, 1p/19q)\n"
                                    response_text += "• Age (more common in adults)\n"
                                    response_text += "• Radiation exposure history\n\n"
                                elif tumor_type == "meningioma":
                                    response_text += "Meningioma Risk Factors:\n"
                                    response_text += "• Female gender (more common)\n"
                                    response_text += "• Age (more common in older adults)\n"
                                    response_text += "• Hormone factors\n"
                                    response_text += "• Radiation exposure\n\n"
                                elif tumor_type == "pituitary":
                                    response_text += "Pituitary Tumor Risk Factors:\n"
                                    response_text += "• Genetic syndromes\n"
                                    response_text += "• Family history\n\n"
                            
                            if confidence_levels:
                                max_conf = 0
                                for confs in confidence_levels.values():
                                    if confs:
                                        max_conf = max(max_conf, max(confs))
                                
                                response_text += "OVERALL RISK ASSESSMENT:\n\n"
                                if max_conf >= 80:
                                    response_text += f"Confidence Level: {max_conf:.1f}% (HIGH) ⚠️⚠️⚠️\n"
                                    response_text += "Risk Status: HIGH RISK - Immediate attention required\n"
                                    response_text += "🚨 Urgency: See doctor within 24-48 hours\n\n"
                                elif max_conf >= 60:
                                    response_text += f"Confidence Level: {max_conf:.1f}% (MODERATE) ⚠️⚠️\n"
                                    response_text += "Risk Status: MODERATE RISK - Medical evaluation needed\n"
                                    response_text += "⚠️ Urgency: Schedule within 1-2 weeks\n\n"
                                else:
                                    response_text += f"Confidence Level: {max_conf:.1f}% (LOW) ⚠️\n"
                                    response_text += "Risk Status: LOW RISK - Further evaluation recommended\n"
                                    response_text += "ℹ️ Urgency: Schedule within 2-4 weeks\n\n"
                            else:
                                response_text += "OVERALL RISK ASSESSMENT:\n\n"
                                response_text += "Risk Status: Tumor detected - Medical consultation recommended\n"
                                response_text += "Urgency: Schedule appointment for evaluation\n\n"
                        
                        # Detailed Explanation
                        if report_analysis.get('detailed_explanation'):
                            detailed = report_analysis.get('detailed_explanation', '')
                            if "Glioma Detected" in detailed:
                                response_text += "Detailed Explanation:\n\n"
                                response_text += "Glioma: Primary brain tumor from glial cells\n"
                                response_text += "• Can be low-grade or high-grade\n"
                                response_text += "• Treatment: Surgery, radiation, chemotherapy\n\n"
                            elif "Meningioma Detected" in detailed:
                                response_text += "Detailed Explanation:\n\n"
                                response_text += "Meningioma: Tumor from protective brain layers\n"
                                response_text += "• Usually benign, slow-growing\n"
                                response_text += "• Treatment: Observation, surgery, or radiation\n\n"
                            elif "Pituitary Tumor Detected" in detailed:
                                response_text += "Detailed Explanation:\n\n"
                                response_text += "Pituitary Tumor: Growth in pituitary gland\n"
                                response_text += "• Usually benign\n"
                                response_text += "• Treatment: Surgery, medication, or radiation\n\n"
                        
                        # Urgency Recommendation
                        urgency_rec = report_analysis.get('urgency_recommendation', {})
                        if urgency_rec:
                            response_text += "🚨 URGENCY RECOMMENDATION 🚨\n\n"
                            msg = urgency_rec.get('message', '')
                            if "IMMEDIATE" in msg:
                                response_text += "IMMEDIATE ACTION REQUIRED\n"
                                response_text += "• Contact healthcare provider WITHIN 24-48 HOURS\n"
                                response_text += "• If severe symptoms, go to emergency department IMMEDIATELY\n"
                                response_text += "• Schedule urgent appointment with specialist\n\n"
                            elif "URGENT" in msg:
                                response_text += "URGENT ACTION REQUIRED\n"
                                response_text += "• Schedule appointment WITHIN 1-2 WEEKS\n"
                                response_text += "• Request referral to specialist\n"
                                response_text += "• Monitor symptoms closely\n\n"
                            else:
                                response_text += "SCHEDULED EVALUATION RECOMMENDED\n"
                                response_text += "• Schedule appointment within 2-4 weeks\n\n"
                            response_text += f"Timeframe: {urgency_rec.get('timeframe', 'Consult with your doctor')}\n\n"
                        
                        # Treatment Suggestions
                        if report_analysis.get('treatment_suggestions'):
                            response_text += "💊 Treatment Options:\n\n"
                            suggestions = report_analysis.get('treatment_suggestions', '')
                            if "For Glioma" in suggestions:
                                response_text += "Glioma Treatment:\n"
                                response_text += "• Surgery - Maximal safe resection\n"
                                response_text += "• Radiation Therapy\n"
                                response_text += "• Chemotherapy\n"
                                response_text += "• Targeted Therapy\n\n"
                            elif "For Meningioma" in suggestions:
                                response_text += "Meningioma Treatment:\n"
                                response_text += "• Observation - For small tumors\n"
                                response_text += "• Surgery - When feasible\n"
                                response_text += "• Stereotactic Radiosurgery\n"
                                response_text += "• Fractionated Radiation\n\n"
                            elif "For Pituitary" in suggestions:
                                response_text += "Pituitary Treatment:\n"
                                response_text += "• Surgery - Transsphenoidal resection\n"
                                response_text += "• Medical Therapy\n"
                                response_text += "• Hormone Replacement\n\n"
                        
                        response_text += "Need more information? Ask me about:\n"
                        response_text += "• Specific findings\n"
                        response_text += "• Treatment details\n"
                        response_text += "• Next steps\n"
                    else:
                        response_text += "Report uploaded successfully. Analysis is being processed.\n\n"
                        response_text += "You can ask me about:\n• Risk factors\n• Findings\n• Treatment options\n• Medical terminology\n• Next steps\n\nWhat would you like to know?"
            
            # Default response if no match found and no report
            elif not response_text:
                response_text = "I understand you're asking about brain tumors. I can provide information about:\n\n• Symptoms and warning signs\n• Diagnosis procedures\n• Treatment options\n• Types of brain tumors (glioma, meningioma, pituitary)\n• Recovery process\n• Causes and risk factors\n\nCould you please be more specific about what you'd like to know? For example, you could ask about 'symptoms', 'treatment', 'types', or ask about a specific tumor type like 'glioma'."
            
            return jsonify({
                "response": response_text,
                "timestamp": datetime.utcnow().isoformat(),
                "report_id": report_id if report_id else None
            })
        except Exception as e:
            logger.exception("Error in handle_chat")
            return jsonify({
                "error": f"An error occurred: {str(e)}",
                "response": "I apologize, but I encountered an error processing your request. Please try again or rephrase your question."
            }), 500

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded under form field 'image'"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save upload to a temp file
        filename = f"upload_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        upload_path = os.path.join(GRADCAM_DIR, filename)  # reuse same dir; it's served and ignored typically
        file.save(upload_path)
        logger.info(f"Upload saved to {upload_path}")

        try:
            # If utils exposes functions that accept a preloaded model, try that first
            if predict_utils is not None and hasattr(predict_utils, "predict_with_model"):
                logger.info("Invoking utils.predict_with_model …")
                pred_label, confidences = predict_utils.predict_with_model(  # type: ignore
                    image_path=upload_path, class_names=class_names, model=model_handle
                )
                if hasattr(predict_utils, "generate_gradcam_with_model"):
                    logger.info("Invoking utils.generate_gradcam_with_model …")
                    gradcam_obj = predict_utils.generate_gradcam_with_model(  # type: ignore
                        image_path=upload_path, model=model_handle, output_dir=GRADCAM_DIR
                    )
                else:
                    gradcam_obj = None
            else:
                logger.info("Invoking try_predict_with_utils …")
                pred_label, confidences, gradcam_obj = try_predict_with_utils(upload_path, class_names)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        # gradcam_obj may now be a dict with overlay/mask/bbox
        gradcam_overlay_path = None
        gradcam_mask_path = None
        gradcam_bbox = None
        if isinstance(gradcam_obj, dict):
            gradcam_overlay_path = gradcam_obj.get("overlay_path")
            gradcam_mask_path = gradcam_obj.get("mask_path")
            gradcam_bbox = gradcam_obj.get("bbox")
            gradcam_center = gradcam_obj.get("center")
            gradcam_path = gradcam_overlay_path or gradcam_mask_path
        else:
            gradcam_path = ensure_gradcam_path(gradcam_obj) if gradcam_obj is not None else None
            gradcam_center = None
        # Convert absolute path to URL under /static, only if the path exists
        gradcam_url = None
        gradcam_mask_url = None
        try:
            if gradcam_path and os.path.exists(gradcam_path):
                rel_gradcam = os.path.relpath(gradcam_path, STATIC_DIR).replace("\\", "/")
                gradcam_url = f"/static/{rel_gradcam}"
        except Exception:
            gradcam_url = None
        if gradcam_mask_path:
            try:
                if os.path.exists(gradcam_mask_path):
                    rel_mask = os.path.relpath(gradcam_mask_path, STATIC_DIR).replace("\\", "/")
                    gradcam_mask_url = f"/static/{rel_mask}"
            except Exception:
                gradcam_mask_url = None
        if gradcam_path:
            logger.info(f"Grad-CAM saved to {gradcam_path}")

        # Original uploaded image URL (we saved it under GRADCAM_DIR)
        rel_original = os.path.relpath(upload_path, STATIC_DIR).replace("\\", "/")
        original_url = f"/static/{rel_original}"

        # Normalize confidences to JSON-safe dict
        safe_confidences = normalize_confidences(confidences, class_names)
        # Debug: log per-class confidences sorted desc for troubleshooting
        try:
            sorted_conf = sorted(safe_confidences.items(), key=lambda kv: kv[1], reverse=True)
            logger.info("Confidences: " + ", ".join([f"{k}={v:.3f}" for k, v in sorted_conf]))
        except Exception:
            pass
        primary_conf = float(safe_confidences.get(pred_label, 0.0))

        suggestion = build_treatment_suggestion(pred_label)
        diagnostic = human_readable_diagnostic(pred_label, primary_conf)
        # If predicted no_tumor: do not show localization or Grad-CAM
        if pred_label == "no_tumor":
            localization = "NIL"
            gradcam_url = None
            gradcam_mask_url = None
            gradcam_bbox = None
        else:
            # Prefer precise localization from hotspot center if provided
            localization = "undetermined"
            try:
                if gradcam_center and gradcam_path and os.path.exists(gradcam_path):
                    from PIL import Image  # lazy import
                    cx, cy = gradcam_center if isinstance(gradcam_center, (list, tuple)) else (None, None)
                    img = Image.open(gradcam_path)
                    w, h = img.size
                    if isinstance(cx, int) and isinstance(cy, int) and w > 0 and h > 0:
                        hemisphere = "left" if cx < w/2 else "right"
                        y_rel = cy / float(h)
                        if y_rel < 0.33:
                            lobe = "frontal or parietal (superior)"
                        elif y_rel < 0.66:
                            lobe = "parietal or temporal (mid)"
                        else:
                            lobe = "temporal or occipital (inferior/posterior)"
                        localization = f"{lobe}, {hemisphere} hemisphere"
                    else:
                        localization = estimate_lobe_from_gradcam(gradcam_path)
                else:
                    localization = estimate_lobe_from_gradcam(gradcam_path)
            except Exception:
                localization = estimate_lobe_from_gradcam(gradcam_path)

        # If an LLM API key is configured, refine treatment suggestion via LLM (Groq preferred, then Grok)
        try:
            loc_text = localization if pred_label != "no_tumor" and localization else ""
            prompt_ts = (
                "Provide a concise, evidence-aligned treatment plan for the suspected brain tumor. "
                "Return 2-3 sentences, no bullets, no prognosis. Keep under 60 words.\n\n"
                f"Tumor type: {pred_label}\n"
                f"Confidence: {primary_conf:.3f}\n"
                f"Estimated location: {loc_text}\n"
                f"Baseline plan: {suggestion}"
            )
            llm_ts = ""
            if os.environ.get("GROQ_API_KEY", "").strip():
                llm_ts = _call_groq_api(prompt_ts).strip()
            elif os.environ.get("GROK_API_KEY", "").strip():
                llm_ts = _call_grok_api(prompt_ts).strip()
            if llm_ts:
                suggestion = llm_ts
        except Exception:
            pass

        # Probe overlay image size for frontend auto-view logic
        gradcam_size = None
        try:
            if gradcam_path and os.path.exists(gradcam_path):
                from PIL import Image as _Image
                _im = _Image.open(gradcam_path)
                gradcam_size = (int(_im.size[0]), int(_im.size[1]))
        except Exception:
            gradcam_size = None

        response = {
            "prediction": pred_label,
            "confidence": primary_conf,
            "confidences": safe_confidences,
            "gradcam_url": gradcam_url,
            "treatment_suggestion": suggestion,
            "diagnostic_text": diagnostic,
            "localization": localization,
            "highlights": build_highlight_tokens(pred_label, localization),
            "bbox": gradcam_bbox,
            "gradcam_mask_url": gradcam_mask_url,
            "original_url": original_url,
            "hotspot_center": gradcam_center,
            "gradcam_size": gradcam_size,
        }
        logger.info(f"Responding with prediction={pred_label}, confidence={primary_conf:.3f}")
        
        # Automatically save analysis result as a patient record
        try:
            patient_id = str(uuid.uuid4())[:8]
            analysis_patient = {
                "id": patient_id,
                "name": "Analysis Patient",
                "age": "",
                "condition": pred_label if pred_label != "no_tumor" else "No Tumor Detected",
                "lastVisit": datetime.utcnow().strftime("%Y-%m-%d"),
                "report": None,  # Will be updated when report is generated
                "imagePath": original_url,
                "status": "active",
                "confidence": primary_conf,
                "localization": localization,
                "analysisDate": datetime.utcnow().isoformat(),
                "gradcamUrl": gradcam_url,
                "created_at": datetime.utcnow().isoformat()
            }
            patients.append(analysis_patient)
            save_patients(patients)  # Save to persistent storage
            response["patient_id"] = patient_id
        except Exception as e:
            logger.warning(f"Failed to save analysis result: {e}")
        
        return jsonify(response)

    def _call_grok_api(prompt: str) -> str:
        api_key = os.environ.get("GROK_API_KEY", "").strip()
        if not api_key:
            return ""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are a clinical report assistant. Keep language clear and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text or ""
        except Exception as e:
            logger.warning(f"Grok API call failed: {e}")
            return ""

    def _call_groq_api(prompt: str) -> str:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return ""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                # Common Groq chat completions endpoint (OpenAI-compatible)
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a clinical report assistant. Keep language clear and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text or ""
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
            return ""

    def _build_pdf(report_data: dict) -> str:
        # Create a formatted PDF report using Platypus with proper wrapping
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib import colors
        except Exception as e:
            raise RuntimeError("Report generation requires the 'reportlab' package. Install with: pip install reportlab") from e

        import unicodedata

        def norm(text: str) -> str:
            if not isinstance(text, str):
                return ""
            # Replace special dashes/bullets and normalize to ASCII to avoid font glyph issues
            text = text.replace("–", "-").replace("—", "-").replace("•", "-")
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

        filename = f"report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.pdf"
        out_path = os.path.join(REPORTS_DIR, filename)

        doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        title = ParagraphStyle('title', parent=styles['Heading1'], alignment=TA_LEFT, spaceAfter=12)
        h2 = ParagraphStyle('h2', parent=styles['Heading2'], spaceBefore=12, spaceAfter=6)
        body = ParagraphStyle('body', parent=styles['BodyText'], leading=14, spaceAfter=6)

        elems = []
        elems.append(Paragraph("Brain Tumor Analysis Report", title))
        patient_name = report_data.get('patient_name') or ''
        patient_phone = report_data.get('patient_phone') or ''
        if patient_name:
            elems.append(Paragraph(norm(f"Patient: {patient_name}"), styles['BodyText']))
        if patient_phone:
            elems.append(Paragraph(norm(f"Phone: {patient_phone}"), styles['BodyText']))
        elems.append(Paragraph(norm(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"), styles['BodyText']))
        elems.append(Spacer(1, 6))

        # Summary
        elems.append(Paragraph("SUMMARY", h2))
        elems.append(Paragraph(norm(f"Tumor type: {report_data.get('prediction','-')}"), body))
        elems.append(Paragraph(norm(f"Confidence: {(report_data.get('confidence',0.0)*100):.1f}%"), body))
        loc = report_data.get("localization") or "NIL"
        elems.append(Paragraph(norm(f"Estimated location: {loc}"), body))
        elems.append(Spacer(1, 6))

        narrative = report_data.get("narrative") or report_data.get("diagnostic_text") or ""
        if narrative:
            elems.append(Paragraph("DIAGNOSTIC NOTE", h2))
            elems.append(Paragraph(norm(narrative), body))
            elems.append(Spacer(1, 6))

        treatment = report_data.get("treatment_suggestion") or ""
        if treatment:
            elems.append(Paragraph("TREATMENT SUGGESTION", h2))
            elems.append(Paragraph(norm(treatment), body))
            elems.append(Spacer(1, 6))

        # Images side-by-side if available
        from urllib.parse import urlparse
        def to_path(url: str) -> str:
            if not url:
                return ""
            u = url
            # Strip origin if absolute URL
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    u = parsed.path or url
            except Exception:
                u = url
            if u.startswith("/static/"):
                rel = u.replace("/static/", "").lstrip("/")
                return os.path.join(STATIC_DIR, rel.replace("/", os.sep))
            # Fallback: treat as path relative to app root
            return os.path.join(APP_ROOT, u.lstrip("/"))

        gradcam_url = report_data.get("gradcam_url")
        orig_url = report_data.get("original_url")
        left_path = to_path(orig_url)
        right_path = to_path(gradcam_url)
        row = []
        img_w = 240
        img_h = 180
        try:
            if left_path and os.path.exists(left_path):
                row.append(RLImage(left_path, width=img_w, height=img_h))
            if right_path and os.path.exists(right_path):
                row.append(RLImage(right_path, width=img_w, height=img_h))
            if row:
                tbl = Table([row], colWidths=[img_w] * len(row))
                tbl.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BOX', (0,0), (-1,-1), 0.25, colors.grey),
                ]))
                elems.append(Spacer(1, 6))
                elems.append(tbl)
        except Exception as e:
            elems.append(Paragraph(norm(f"[Image unavailable: {e}]"), styles['BodyText']))

        doc.build(elems)
        return out_path

    @app.route("/report", methods=["POST"])
    def generate_report():
        data = request.get_json(silent=True) or {}
        # Optionally call Grok to craft a human-readable narrative
        narrative = ""
        prediction = data.get("prediction", "")
        confidence = data.get("confidence", 0.0)
        localization = data.get("localization", "")
        diagnostic = data.get("diagnostic_text", "")
        suggestion = data.get("treatment_suggestion", "")

        prompt = (
            "Create a short, patient-friendly MRI brain tumor report. Include tumor type, confidence as %," \
            " location (if provided), one-paragraph diagnostic explanation, and concise treatment plan. " \
            "Avoid prognosis. Keep under 120 words.\n\n" \
            f"Tumor type: {prediction}\nConfidence: {confidence:.3f}\nLocation: {localization}\n" \
            f"Diagnostic: {diagnostic}\nTreatment: {suggestion}"
        )
        # Prefer Groq for narrative, then Grok; fall back to diagnostic text
        narrative = ""
        try:
            if os.environ.get("GROQ_API_KEY", "").strip():
                narrative = _call_groq_api(prompt)
            elif os.environ.get("GROK_API_KEY", "").strip():
                narrative = _call_grok_api(prompt)
        except Exception:
            narrative = ""
        if not narrative:
            narrative = diagnostic

        payload = {
            "prediction": prediction,
            "confidence": confidence,
            "localization": localization or "NIL",
            "diagnostic_text": diagnostic,
            "treatment_suggestion": suggestion,
            "gradcam_url": data.get("gradcam_url"),
            "original_url": data.get("original_url"),
            "narrative": narrative,
            "patient_name": data.get("patient_name", ""),
            "patient_phone": data.get("patient_phone", ""),
        }
        try:
            pdf_path = _build_pdf(payload)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        rel = os.path.relpath(pdf_path, STATIC_DIR).replace("\\", "/")
        report_url = f"/static/{rel}"
        
        # Update patient record with report URL if patient_id is provided
        patient_id = data.get("patient_id")
        if patient_id:
            for patient in patients:
                if patient.get("id") == patient_id:
                    patient["report"] = report_url
                    patient_name = data.get("patient_name", patient.get("name", "Analysis Patient"))
                    patient_phone = data.get("patient_phone", patient.get("phone", ""))
                    if patient_name and patient_name != "Analysis Patient":
                        patient["name"] = patient_name
                    if patient_phone:
                        patient["phone"] = patient_phone
                    break
            save_patients(patients)  # Save updated patient data
        
        return jsonify({"report_url": report_url})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


