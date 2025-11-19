import streamlit as st
import numpy as np
from PIL import Image
import time
import cv2

# =============================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================
st.set_page_config(
    page_title="SafeBuild Pro - YOLO",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CSS PERSONALIZADO
# =============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #DC2626;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-medium {
        background-color: #FEF3C7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #D97706;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-ok {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #059669;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .yolo-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# CARGA DEL MODELO YOLO
# =============================================
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO y lo cachea para mejor performance"""
    try:
        from ultralytics import YOLO
        # YOLOv8n es liviano y rápido - perfecto para Streamlit Cloud
        model = YOLO('yolov8n.pt')
        st.sidebar.success("✅ YOLOv8n cargado exitosamente!")
        return model
    except Exception as e:
        st.error(f"❌ Error cargando YOLO: {str(e)}")
        return None

# =============================================
# SISTEMA EXPERTO DE SEGURIDAD
# =============================================
class SafetyExpertSystem:
    def __init__(self):
        self.rules = {
            'no_ppe_high_risk': {
                'condition': lambda stats: stats['persons'] >= 2 and stats['hardhats'] == 0,
                'message': "ALTA: Múltiples trabajadores sin protección en cabeza",
                'level': "ALTA",
                'action': "Suspender actividades y proveer EPP inmediatamente"
            },
            'partial_ppe': {
                'condition': lambda stats: stats['persons'] > 0 and stats['hardhats'] < stats['persons'],
                'message': "MEDIA: Algunos trabajadores sin equipo de protección",
                'level': "MEDIA", 
                'action': "Capacitación inmediata y verificación de EPP"
            },
            'construction_heavy_equipment': {
                'condition': lambda stats: stats['persons'] > 0 and stats['heavy_equipment'] > 0,
                'message': "MEDIA: Personal trabajando cerca de maquinaria pesada",
                'level': "MEDIA",
                'action': "Establecer zonas de seguridad y señalización"
            },
            'safe_conditions': {
                'condition': lambda stats: stats['persons'] > 0 and stats['hardhats'] >= stats['persons'],
                'message': "OK: Condiciones de seguridad adecuadas detectadas",
                'level': "OK",
                'action': "Continuar con monitoreo rutinario"
            },
            'no_activity': {
                'condition': lambda stats: stats['persons'] == 0,
                'message': "INFO: No se detectó actividad en el área",
                'level': "OK", 
                'action': "Monitoreo continuo del área"
            }
        }
    
    def analyze_detections(self, detections):
        """Analiza las detecciones de YOLO y aplica reglas de seguridad"""
        
        # Contar objetos detectados
        stats = {
            'persons': len([d for d in detections if d['class_name'] == 'person']),
            'hardhats': len([d for d in detections if d['class_name'] in ['hard hat', 'safety helmet']]),
            'heavy_equipment': len([d for d in detections if d['class_name'] in ['truck', 'car', 'bus', 'construction equipment']]),
            'total_detections': len(detections)
        }
        
        # Aplicar reglas en orden de prioridad
        for rule_name, rule in self.rules.items():
            if rule['condition'](stats):
                return {
                    'alert_level': rule['level'],
                    'alert_message': rule['message'],
                    'recommended_action': rule['action'],
                    'statistics': stats
                }
        
        return {
            'alert_level': "OK",
            'alert_message': "Situación normal detectada",
            'recommended_action': "Continuar operaciones normales",
            'statistics': stats
        }

# =============================================
# PROCESAMIENTO CON YOLO
# =============================================
def process_image_with_yolo(model, image, confidence_threshold=0.5):
    """Procesa una imagen con YOLO y retorna detecciones"""
    try:
        # Convertir PIL Image a numpy array (RGB)
        image_np = np.array(image)
        
        # Convertir RGB a BGR para OpenCV (YOLO espera BGR)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Ejecutar YOLO
        results = model(image_np, conf=confidence_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    bbox = [float(coord) for coord in box.xyxy[0].tolist()]
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'class_id': class_id
                    })
        
        return detections, results[0] if results else None
        
    except Exception as e:
        st.error(f"Error en procesamiento YOLO: {str(e)}")
        return [], None

def draw_yolo_predictions(image, results):
    """Dibuja las predicciones de YOLO en la imagen"""
    try:
        if results is not None:
            # Usar el método plot de YOLO que ya incluye bounding boxes
            annotated_image = results.plot()
            # Convertir BGR a RGB para Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(annotated_image_rgb)
        return image
    except Exception as e:
        st.warning(f"No se pudieron dibujar las predicciones: {str(e)}")
        return image

# =============================================
# INICIALIZACIÓN
# =============================================
model = load_yolo_model()
expert_system = SafetyExpertSystem()

# =============================================
# INTERFAZ PRINCIPAL
# =============================================
st.markdown('<h1 class="main-header">🦺 SafeBuild con YOLO Real</h1>', unsafe_allow_html=True)
st.markdown("### Sistema Inteligente de Detección de Seguridad en Obras")

# Sidebar
st.sidebar.markdown("### ⚙️ Configuración YOLO")
confidence_threshold = st.sidebar.slider(
    "Umbral de Confianza", 
    0.1, 0.9, 0.5, 0.05,
    help="Ajusta la sensibilidad de las detecciones"
)

st.sidebar.markdown("### 🧠 Información del Modelo")
st.sidebar.markdown('<span class="yolo-badge">YOLOv8 Nano</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="yolo-badge">80 Clases COCO</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="yolo-badge">Detección en Tiempo Real</span>', unsafe_allow_html=True)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["📸 Analizar Imagen", "🎯 Demo Rápida", "ℹ️ Acerca de YOLO"])

with tab1:
    st.subheader("Sube una imagen para análisis con YOLO")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de obra de construcción:",
        type=['jpg', 'jpeg', 'png'],
        help="YOLO detectará personas, vehículos, equipos y más"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen original
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📷 Imagen Original", use_column_width=True)
        
        with col2:
            if st.button("🧠 EJECUTAR YOLO", type="primary", use_container_width=True):
                if model is None:
                    st.error("❌ YOLO no está disponible. Revisa los logs de error.")
                else:
                    with st.spinner("🔍 YOLO analizando imagen..."):
                        # Barra de progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simular etapas de procesamiento
                        stages = [
                            "Cargando modelo YOLO...",
                            "Extrayendo características...", 
                            "Ejecutando detección...",
                            "Procesando resultados..."
                        ]
                        
                        for i, stage in enumerate(stages):
                            progress_bar.progress((i + 1) * 25)
                            status_text.text(f"⏳ {stage}")
                            time.sleep(0.5)
                        
                        # Procesar con YOLO
                        detections, yolo_results = process_image_with_yolo(
                            model, image, confidence_threshold
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Análisis completado!")
                    
                    # Mostrar resultados
                    if yolo_results is not None:
                        annotated_image = draw_yolo_predictions(image, yolo_results)
                        st.image(annotated_image, 
                               caption=f"🎯 Detecciones YOLO ({len(detections)} objetos)", 
                               use_column_width=True)
                    
                    # Análisis de seguridad
                    analysis = expert_system.analyze_detections(detections)
                    stats = analysis['statistics']
                    
                    # Mostrar métricas
                    st.subheader("📊 Métricas de Detección")
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    
                    with col_met1:
                        st.metric("Personas", stats['persons'])
                    with col_met2:
                        st.metric("Cascos/EPP", stats['hardhats'])
                    with col_met3:
                        st.metric("Maquinaria", stats['heavy_equipment'])
                    with col_met4:
                        st.metric("Total Objetos", stats['total_detections'])
                    
                    # Mostrar alerta de seguridad
                    st.subheader("🚨 Análisis de Seguridad")
                    alert_level = analysis['alert_level']
                    
                    if alert_level == "ALTA":
                        st.markdown(f"""
                        <div class="alert-high">
                            <h3>🚨 {analysis['alert_message']}</h3>
                            <p><strong>📋 Acción Recomendada:</strong> {analysis['recommended_action']}</p>
                            <p><strong>🔍 Detectado por:</strong> YOLOv8 + Sistema Experto</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif alert_level == "MEDIA":
                        st.markdown(f"""
                        <div class="alert-medium">
                            <h3>⚠️ {analysis['alert_message']}</h3>
                            <p><strong>📋 Acción Recomendada:</strong> {analysis['recommended_action']}</p>
                            <p><strong>🔍 Detectado por:</strong> YOLOv8 + Sistema Experto</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-ok">
                            <h3>✅ {analysis['alert_message']}</h3>
                            <p><strong>📋 Acción Recomendada:</strong> {analysis['recommended_action']}</p>
                            <p><strong>🔍 Detectado por:</strong> YOLOv8 + Sistema Experto</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detalles de detecciones
                    with st.expander("🔍 Ver detalles de todas las detecciones"):
                        if detections:
                            for i, det in enumerate(detections):
                                st.write(f"**{i+1}. {det['class_name'].upper()}** - Confianza: {det['confidence']:.2f}")
                        else:
                            st.info("No se encontraron detecciones por encima del umbral de confianza")

with tab2:
    st.subheader("🎯 Demo Rápida con YOLO")
    st.info("Prueba el sistema con esta funcionalidad de demo")
    
    demo_option = st.selectbox(
        "Selecciona un escenario de demo:",
        ["🏗️ Obra con múltiples trabajadores", "🚧 Área con maquinaria", "👷 Trabajador individual"]
    )
    
    if st.button("🚀 EJECUTAR DEMO YOLO", type="secondary"):
        if model is None:
            st.error("YOLO no disponible para la demo")
        else:
            # Crear imagen de demo simple
            demo_image = create_demo_image(demo_option)
            
            with st.spinner("Ejecutando demo con YOLO..."):
                detections, yolo_results = process_image_with_yolo(
                    model, demo_image, confidence_threshold
                )
                
                col_demo1, col_demo2 = st.columns(2)
                
                with col_demo1:
                    st.image(demo_image, caption="Escenario de Demo", use_column_width=True)
                
                with col_demo2:
                    if yolo_results is not None:
                        annotated_image = draw_yolo_predictions(demo_image, yolo_results)
                        st.image(annotated_image, caption="Detecciones YOLO", use_column_width=True)
                
                # Análisis rápido
                analysis = expert_system.analyze_detections(detections)
                st.success(f"Demo completada: {analysis['alert_message']}")

with tab3:
    st.subheader("🧠 Acerca de YOLO (You Only Look Once)")
    
    st.markdown("""
    ### 🚀 ¿Qué es YOLO?
    
    YOLO es un algoritmo de detección de objetos en tiempo real que:
    
    - **Velocidad**: Procesa imágenes en milisegundos
    - **Precisión**: Alta tasa de detección en múltiples clases
    - **Eficiencia**: Una sola pasada por la red (You Only Look Once)
    
    ### 🏗️ Arquitectura YOLOv8
    
    **Características técnicas:**
    - **Backbone**: CSPDarknet (Extracción de características)
    - **Neck**: PAN-FPN (Fusión de características multi-escala)  
    - **Head**: Detección con anchor-free
    - **Entrenamiento**: 80 clases del dataset COCO
    
    ### 🎯 Clases Relevantes para Seguridad en Obras
    
    YOLO puede detectar automáticamente:
    - 👥 **Personas** - Trabajadores en la obra
    - 🚗 **Vehículos** - Maquinaria y equipos
    - 🎒 **Objetos** - Equipamiento de obra
    - 🚧 **Señales** - Elementos de seguridad
    
    ### 🔧 Integración en SafeBuild
    
    Este sistema combina:
    1. **YOLOv8** para detección de objetos en tiempo real
    2. **Sistema Experto** para análisis de reglas de seguridad
    3. **Streamlit** para interfaz web interactiva
    """)

# =============================================
# FUNCIONES AUXILIARES
# =============================================
def create_demo_image(scenario):
    """Crea una imagen de demo simple"""
    width, height = 640, 480
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Dibujar según escenario
    if "múltiples" in scenario:
        # Personas
        cv2.rectangle(image, (100, 300), (150, 450), (0, 255, 0), -1)
        cv2.rectangle(image, (300, 280), (350, 430), (0, 255, 0), -1)
        cv2.putText(image, "DEMO: OBRA ACTIVA", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif "maquinaria" in scenario:
        # Vehículo
        cv2.rectangle(image, (200, 200), (400, 300), (255, 0, 0), -1)
        cv2.rectangle(image, (150, 320), (180, 450), (0, 255, 0), -1)
        cv2.putText(image, "DEMO: MAQUINARIA", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Persona individual
        cv2.rectangle(image, (300, 250), (350, 400), (0, 255, 0), -1)
        cv2.putText(image, "DEMO: TRABAJADOR", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return Image.fromarray(image)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>SafeBuild Pro v3.0</strong> - YOLOv8 + Streamlit | Trabajo Práctico Integrador IA</p>
    <p style="font-size: 0.8rem;">🧠 Powered by Ultralytics YOLO | 🎯 Detección en Tiempo Real | 🚀 Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
