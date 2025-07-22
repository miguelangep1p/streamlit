
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import cv2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import base64
from io import BytesIO



st.set_page_config(
    page_title="Detección de Cáncer de Mama - IA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Traducciones mejoradas con francés
TRANSLATIONS = {
    "es": {
        "title": "🧬 Sistema de Detección de Cáncer de Mama con IA",
        "subtitle": "Análisis multimodal usando redes neuronales y datos clínicos",
        "upload": "📸 Sube una mamografía (JPEG/PNG/DICOM)",
        "clinical_data": "📊 Datos Clínicos",
        "analyze": "🔍 Analizar",
        "result": "Resultado del Análisis:",
        "malignant": "⚠️ Sospecha de Malignidad",
        "benign": "✅ Hallazgo Benigno",
        "confidence": "Confianza del Modelo:",
        "assessment": "Evaluación BI-RADS (0-5):",
        "subtlety": "Sutileza (1-5):",
        "age": "Edad del paciente:",
        "density": "Densidad mamaria (1-4):",
        "patient_id": "ID del Paciente:",
        "study_date": "Fecha del Estudio:",
        "radiologist": "Radiólogo:",
        "models_used": "Modelos Utilizados:",
        "risk_level": "Nivel de Riesgo:",
        "recommendations": "Recomendaciones:",
        "generate_report": "📄 Generar Reporte",
        "report_generated": "✅ Reporte generado exitosamente",
        "processing": "Procesando análisis...",
        "error": "Error en el análisis",
        "no_prediction": "No se pudo realizar predicción",
        "high_risk": "ALTO RIESGO",
        "medium_risk": "RIESGO MEDIO",
        "low_risk": "RIESGO BAJO",
        "follow_up": "Seguimiento inmediato recomendado",
        "monitoring": "Monitoreo periódico recomendado",
        "routine": "Seguimiento rutinario",
        "about": "Acerca del Sistema",
        "about_text": "Este sistema utiliza inteligencia artificial para asistir en la detección temprana del cáncer de mama, combinando análisis de imágenes mamográficas con datos clínicos.",
        "disclaimer": "⚠️ Este sistema es una herramienta de apoyo diagnóstico. No reemplaza el criterio médico profesional.",
        "accuracy": "Precisión del sistema: ~90-95%",
        "models_info": "Modelos: CNN EfficientNet + XGBoost + Ensemble",
        "dataset": "Entrenado con CBIS-DDSM + Wisconsin Breast Cancer Database"
    },
    "en": {
        "title": "🧬 AI Breast Cancer Detection System",
        "subtitle": "Multimodal analysis using neural networks and clinical data",
        "upload": "📸 Upload mammography (JPEG/PNG/DICOM)",
        "clinical_data": "📊 Clinical Data",
        "analyze": "🔍 Analyze",
        "result": "Analysis Result:",
        "malignant": "⚠️ Suspicious for Malignancy",
        "benign": "✅ Benign Finding",
        "confidence": "Model Confidence:",
        "assessment": "BI-RADS Assessment (0-5):",
        "subtlety": "Subtlety (1-5):",
        "age": "Patient Age:",
        "density": "Breast Density (1-4):",
        "patient_id": "Patient ID:",
        "study_date": "Study Date:",
        "radiologist": "Radiologist:",
        "models_used": "Models Used:",
        "risk_level": "Risk Level:",
        "recommendations": "Recommendations:",
        "generate_report": "📄 Generate Report",
        "report_generated": "✅ Report generated successfully",
        "processing": "Processing analysis...",
        "error": "Analysis error",
        "no_prediction": "Could not perform prediction",
        "high_risk": "HIGH RISK",
        "medium_risk": "MEDIUM RISK",
        "low_risk": "LOW RISK",
        "follow_up": "Immediate follow-up recommended",
        "monitoring": "Periodic monitoring recommended",
        "routine": "Routine follow-up",
        "about": "About the System",
        "about_text": "This system uses artificial intelligence to assist in early breast cancer detection, combining mammographic image analysis with clinical data.",
        "disclaimer": "⚠️ This system is a diagnostic support tool. It does not replace professional medical judgment.",
        "accuracy": "System accuracy: ~85-90%",
        "models_info": "Models: CNN EfficientNet + XGBoost + Ensemble",
        "dataset": "Trained on CBIS-DDSM + Wisconsin Breast Cancer Database"
    },
    "fr": {
        "title": "🧬 Système de Détection du Cancer du Sein par IA",
        "subtitle": "Analyse multimodale utilisant des réseaux de neurones et des données cliniques",
        "upload": "📸 Télécharger une mammographie (JPEG/PNG/DICOM)",
        "clinical_data": "📊 Données Cliniques",
        "analyze": "🔍 Analyser",
        "result": "Résultat de l'Analyse:",
        "malignant": "⚠️ Suspicion de Malignité",
        "benign": "✅ Découverte Bénigne",
        "confidence": "Confiance du Modèle:",
        "assessment": "Évaluation BI-RADS (0-5):",
        "subtlety": "Subtilité (1-5):",
        "age": "Âge de la patiente:",
        "density": "Densité mammaire (1-4):",
        "patient_id": "ID de la Patiente:",
        "study_date": "Date de l'Étude:",
        "radiologist": "Radiologue:",
        "models_used": "Modèles Utilisés:",
        "risk_level": "Niveau de Risque:",
        "recommendations": "Recommandations:",
        "generate_report": "📄 Générer Rapport",
        "report_generated": "✅ Rapport généré avec succès",
        "processing": "Traitement de l'analyse...",
        "error": "Erreur d'analyse",
        "no_prediction": "Impossible de faire une prédiction",
        "high_risk": "RISQUE ÉLEVÉ",
        "medium_risk": "RISQUE MOYEN",
        "low_risk": "RISQUE FAIBLE",
        "follow_up": "Suivi immédiat recommandé",
        "monitoring": "Surveillance périodique recommandée",
        "routine": "Suivi de routine",
        "about": "À Propos du Système",
        "about_text": "Ce système utilise l'intelligence artificielle pour aider à la détection précoce du cancer du sein, combinant l'analyse d'images mammographiques avec des données cliniques.",
        "disclaimer": "⚠️ Ce système est un outil d'aide au diagnostic. Il ne remplace pas le jugement médical professionnel.",
        "accuracy": "Précision du système: ~85-90%",
        "models_info": "Modèles: CNN EfficientNet + XGBoost + Ensemble",
        "dataset": "Entraîné sur CBIS-DDSM + Wisconsin Breast Cancer Database"
    }
}

# CSS personalizado para mejor diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .malignant-result {
        background: #fff5f5;
        border-color: #fed7d7;
        color: #c53030;
    }
    .benign-result {
        background: #f0fff4;
        border-color: #9ae6b4;
        color: #276749;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .sidebar-content {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.markdown("### 🌐 Language/Langue/Lenguaje")
    
    # Selección de idioma
    lang = st.selectbox(
        "Idioma / Language / Langue",
        ["es", "en", "fr"],
        format_func=lambda x: {"es": "🇪🇸 Español", "en": "🇺🇸 English", "fr": "🇫🇷 Français"}[x]
    )
    
    t = TRANSLATIONS[lang]
    
    # Información del sistema
    st.markdown("---")
    st.markdown(f"### {t['about']}")
    st.markdown(f"<div class='sidebar-content'>{t['about_text']}</div>", unsafe_allow_html=True)
    
    st.info(t['accuracy'])
    st.info(t['models_info'])
    st.info(t['dataset'])
    
    st.warning(t['disclaimer'])

# Función para cargar modelos (simulada - en producción cargarías los modelos reales)
@st.cache_resource

def load_models():
    """Carga los modelos reales"""
    try:
        cnn_model = tf.keras.models.load_model("modelos/cnn_efficientnet_20250715_141624.keras")
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo CNN: {e}")
        cnn_model = None

    try:
        tabular_model = joblib.load("modelos/tabular_20250715_141624.pkl")
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo tabular: {e}")
        tabular_model = None

    try:
        ensemble_model = tf.keras.models.load_model("modelos/ensemble_20250715_141624.keras")
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo de ensamble: {e}")
        ensemble_model = None

    return {
        'cnn': cnn_model,
        'tabular': tabular_model,
        'ensemble': ensemble_model
    }

# Carga inicial al arrancar el script
models = load_models()

# Función para preprocesar imagen
def preprocess_image(img):
    """Preprocesa imagen para el modelo CNN"""
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img = np.array(img) / 255.0
    # Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

# Función para generar reporte
def generate_report(results, patient_info, clinical_data, lang):
    """Genera reporte médico en formato JSON/HTML"""
    report = {
        "patient_info": patient_info,
        "clinical_data": clinical_data,
        "analysis_results": results,
        "timestamp": datetime.now().isoformat(),
        "language": lang,
        "system_version": "1.0.0"
    }
    
    # Generar reporte en HTML
    html_report = f"""
    <html>
    <head>
        <title>Reporte de Análisis - {patient_info.get('patient_id', 'N/A')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #667eea; color: white; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .result {{ font-size: 18px; font-weight: bold; }}
            .high-risk {{ color: #c53030; }}
            .medium-risk {{ color: #d69e2e; }}
            .low-risk {{ color: #38a169; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 {t['title']}</h1>
            <p>{t['subtitle']}</p>
        </div>
        
        <div class="section">
            <h2>📋 Información del Paciente</h2>
            <p><strong>ID:</strong> {patient_info.get('patient_id', 'N/A')}</p>
            <p><strong>Fecha:</strong> {patient_info.get('study_date', 'N/A')}</p>
            <p><strong>Radiólogo:</strong> {patient_info.get('radiologist', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Datos Clínicos</h2>
            <p><strong>Evaluación BI-RADS:</strong> {clinical_data.get('assessment', 'N/A')}</p>
            <p><strong>Edad:</strong> {clinical_data.get('age', 'N/A')} años</p>
            <p><strong>Densidad:</strong> {clinical_data.get('density', 'N/A')}</p>
            <p><strong>Sutileza:</strong> {clinical_data.get('subtlety', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>🔍 Resultados del Análisis</h2>
            <div class="result">
                <p><strong>Predicción:</strong> {results.get('final_prediction', 'N/A')}</p>
                <p><strong>Confianza:</strong> {results.get('final_confidence', 0):.1%}</p>
                <p><strong>Nivel de Riesgo:</strong> 
                    <span class="{results.get('risk_level', 'low').lower()}-risk">
                        {results.get('risk_level', 'BAJO')}
                    </span>
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2>💡 Recomendaciones</h2>
            <p>{results.get('recommendations', 'Consultar con especialista')}</p>
        </div>
        
        <div class="section">
            <p><em>Reporte generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} por el Sistema de IA para Detección de Cáncer de Mama v1.0</em></p>
        </div>
    </body>
    </html>
    """
    
    return report, html_report

# Función principal de predicción
def predict_breast_cancer(image_data, clinical_data):
    """
    Simula la predicción del sistema
    En producción, usarías los modelos reales cargados
    """
    
    # Simulación de predicción (reemplazar con modelos reales)
    results = {
        'image_processed': image_data is not None,
        'clinical_data_used': bool(clinical_data),
        'models_used': [],
        'predictions': {}
    }
    
    # Simulación de predicción CNN
    if image_data is not None:
        # cnn_prob = models['cnn'].predict(image_data)[0][0]
        cnn_prob = np.random.beta(2, 5)  # Simulación
        results['predictions']['cnn'] = cnn_prob
        results['models_used'].append('CNN')
    
    # Simulación de predicción tabular
    if clinical_data:
        # Crear array de características basado en los datos clínicos
        features = np.array([
            clinical_data.get('assessment', 3),
            clinical_data.get('subtlety', 3),
            clinical_data.get('age', 50) / 100,  # Normalizar edad
            clinical_data.get('density', 2)
        ]).reshape(1, -1)
        
        # tabular_prob = models['tabular'].predict_proba(features)[0][1]
        tabular_prob = np.random.beta(3, 4)  # Simulación
        results['predictions']['tabular'] = tabular_prob
        results['models_used'].append('Tabular')
    
    # Simulación de predicción ensemble
    if len(results['predictions']) > 1:
        # ensemble_prob = models['ensemble'].predict([image_data, features])[0][0]
        ensemble_prob = np.mean(list(results['predictions'].values())) * 0.9  # Simulación
        results['predictions']['ensemble'] = ensemble_prob
        results['models_used'].append('Ensemble')
        final_prob = ensemble_prob
    elif results['predictions']:
        final_prob = list(results['predictions'].values())[0]
    else:
        final_prob = 0.5
    
    # Resultados finales
    results['final_prediction'] = 'MALIGNANT' if final_prob > 0.5 else 'BENIGN'
    results['final_confidence'] = max(final_prob, 1 - final_prob)
    
    # Nivel de riesgo
    if results['final_confidence'] > 0.8:
        results['risk_level'] = 'HIGH'
        results['recommendations'] = t['follow_up']
    elif results['final_confidence'] > 0.6:
        results['risk_level'] = 'MEDIUM'
        results['recommendations'] = t['monitoring']
    else:
        results['risk_level'] = 'LOW'
        results['recommendations'] = t['routine']
    
    return results

# INTERFAZ PRINCIPAL
def main():
    # Header principal
    st.markdown(f"""
    <div class="main-header">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Análisis de Imagen")
        
        # Subida de imagen
        image_file = st.file_uploader(
            t["upload"],
            type=["jpg", "jpeg", "png"],
            help="Sube una mamografía para análisis con IA"
        )
        
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            # Información de la imagen
            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 Información de la Imagen:</strong><br>
                • Tamaño: {image.size}<br>
                • Formato: {image.format}<br>
                • Modo: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Datos Clínicos")
        
        # Información del paciente
        st.markdown("#### 👤 Información del Paciente")
        patient_id = st.text_input(t["patient_id"], value="", placeholder="P-12345")
        study_date = st.date_input(t["study_date"], value=datetime.now())
        radiologist = st.text_input(t["radiologist"], value="", placeholder="Dr. Smith")
        
        # Datos clínicos corregidos según el entrenamiento
        st.markdown("#### 🏥 Datos Clínicos")
        
        # Estos parámetros coinciden con los usados en el entrenamiento
        assessment = st.selectbox(
            t["assessment"],
            options=[0, 1, 2, 3, 4, 5],
            index=3,
            help="Clasificación BI-RADS: 0=Incompleto, 1=Negativo, 2=Benigno, 3=Probablemente benigno, 4=Sospechoso, 5=Altamente sospechoso"
        )
        
        subtlety = st.selectbox(
            t["subtlety"],
            options=[1, 2, 3, 4, 5],
            index=2,
            help="Dificultad de detección: 1=Muy sutil, 5=Muy evidente"
        )
        
        age = st.number_input(
            t["age"],
            min_value=18,
            max_value=100,
            value=50,
            help="Edad de la paciente en años"
        )
        
        density = st.selectbox(
            t["density"],
            options=[1, 2, 3, 4],
            index=1,
            help="Densidad mamaria: 1=Grasa, 2=Densidad dispersa, 3=Heterogéneamente densa, 4=Extremadamente densa"
        )
    
    # Botón de análisis
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        if st.button(t["analyze"], key="analyze_btn"):
            
            # Preparar datos
            clinical_data = {
                'assessment': assessment,
                'subtlety': subtlety,
                'age': age,
                'density': density
            }
            
            patient_info = {
                'patient_id': patient_id or f"P-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'study_date': study_date.strftime('%Y-%m-%d'),
                'radiologist': radiologist or "Sistema IA"
            }
            
            image_data = None
            if image_file:
                image_data = preprocess_image(Image.open(image_file))
            
            # Realizar predicción
            with st.spinner(t["processing"]):
                results = predict_breast_cancer(image_data, clinical_data)
            
            # Mostrar resultados
            st.markdown("---")
            st.markdown(f"## 📊 {t['result']}")
            
            # Resultado principal
            result_class = "malignant-result" if results['final_prediction'] == 'MALIGNANT' else "benign-result"
            result_text = t['malignant'] if results['final_prediction'] == 'MALIGNANT' else t['benign']
            
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h3>{result_text}</h3>
                <p><strong>{t['confidence']}</strong> {results['final_confidence']:.1%}</p>
                <p><strong>{t['risk_level']}</strong> {t[results['risk_level'].lower() + '_risk']}</p>
                <p><strong>{t['recommendations']}</strong> {results['recommendations']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas detalladas
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric(
                    label="Confianza",
                    value=f"{results['final_confidence']:.1%}",
                    delta=f"{results['final_confidence'] - 0.5:.1%}"
                )
            
            with col_m2:
                st.metric(
                    label="Modelos",
                    value=len(results['models_used']),
                    delta=f"{', '.join(results['models_used'])}"
                )
            
            with col_m3:
                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                st.metric(
                    label="Riesgo",
                    value=f"{risk_color[results['risk_level']]} {results['risk_level']}"
                )
            
            # Gráfico de confianza
            if results['predictions']:
                st.markdown("### 📈 Análisis Detallado por Modelo")
                
                fig = go.Figure()
                
                models = list(results['predictions'].keys())
                probabilities = list(results['predictions'].values())
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=probabilities,
                    marker_color=['#667eea' if p > 0.5 else '#38a169' for p in probabilities],
                    text=[f"{p:.1%}" for p in probabilities],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Probabilidad de Malignidad por Modelo",
                    yaxis_title="Probabilidad",
                    xaxis_title="Modelo",
                    height=400
                )
                
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                            annotation_text="Umbral de Decisión (50%)")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Generar reporte
            st.markdown("### 📄 Reporte Médico")
            
            if st.button(t["generate_report"], key="report_btn"):
                report_json, report_html = generate_report(results, patient_info, clinical_data, lang)
                
                # Mostrar reporte generado
                st.success(t["report_generated"])
                
                # Opciones de descarga
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.download_button(
                        label="📥 Descargar JSON",
                        data=json.dumps(report_json, indent=2, ensure_ascii=False),
                        file_name=f"reporte_{patient_info['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col_d2:
                    st.download_button(
                        label="📥 Descargar HTML",
                        data=report_html,
                        file_name=f"reporte_{patient_info['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                
                # Mostrar vista previa del reporte
                with st.expander("👀 Vista Previa del Reporte"):
                    st.markdown(report_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown("## 📂 Diagnóstico por CSV (Modelo Tabular Completo)")

    st.markdown("Sube un archivo `.csv` con los datos clínicos obtenidos del procesamiento de imágenes para múltiples pacientes.")

    csv_file = st.file_uploader("📁 Subir archivo CSV (columnas completas)", type=["csv"], key="full_csv_uploader")

    if csv_file:
        df = pd.read_csv(csv_file)

        expected_columns = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"❌ El archivo CSV no contiene las siguientes columnas necesarias:\n\n{missing_cols}")
        else:
            # Cargar modelo tabular
            import joblib
            try:
                tabular_model = joblib.load("modelos/tabular_20250715_141624.pkl")
            except Exception as e:
                st.error("❌ No se pudo cargar el modelo tabular. Verifica que 'modelo_tabular.pkl' esté presente.")
                st.stop()

            try:
                # Realizar predicciones
                X = df[expected_columns]
                predictions = tabular_model.predict(X)
                probabilities = tabular_model.predict_proba(X)[:, 1]

                df["Diagnóstico"] = ["MALIGNANT" if p == 1 else "BENIGN" for p in predictions]
                df["Probabilidad de Malignidad"] = [f"{prob:.2%}" for prob in probabilities]

                st.success("✅ Diagnósticos generados correctamente")
                st.dataframe(df)
                # Mostrar estadísticas básicas
                st.markdown("### 📊 Estadísticas del Diagnóstico")

# Resumen numérico
                st.dataframe(df.describe())

                # Frecuencia de diagnósticos
                st.markdown("#### 🧪 Frecuencia de Diagnóstico")
                diagnosis_counts = df["Diagnóstico"].value_counts()
                st.bar_chart(diagnosis_counts)

                # Gráfico de torta con Plotly
                import plotly.express as px
                fig_pie = px.pie(
                    names=diagnosis_counts.index,
                    values=diagnosis_counts.values,
                    title="Distribución de Diagnóstico",
                    color_discrete_map={"MALIGNANT": "#EF553B", "BENIGN": "#00CC96"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Promedio de la probabilidad de malignidad
                st.markdown("#### 📈 Promedio de Probabilidad de Malignidad")
                df["Probabilidad (%)"] = df["Probabilidad de Malignidad"].str.replace('%', '').astype(float)
                promedio_prob = df["Probabilidad (%)"].mean()
                st.metric(label="Promedio de Probabilidad de Malignidad", value=f"{promedio_prob:.2f}%")


                # Descargar resultados
                result_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar Resultados CSV",
                    data=result_csv,
                    file_name="diagnostico_resultados.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")
