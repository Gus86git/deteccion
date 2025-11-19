import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import time
from ultralytics import YOLO
import tempfile
import os

# =============================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================
st.set_page_config(
    page_title="SafeBuild - Monitoreo de Seguridad con IA",
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
        font-weight: bold;
    }
    .alert-high {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #DC2626;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-medium {
        background-color: #FEF3C7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #D97706;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-ok {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #059669;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #E2E8F0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .detection-box {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1E40AF;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1E3A8A;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SISTEMA EXPERTO DE SEGURIDAD
# =============================================
class SafetyExpertSystem:
    def __init__(self):
        self.rules = {
            'no_helmet_critical': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] == 0,
                'message': "CRÍTICO: Ningún trabajador usa casco de seguridad",
                'level': "ALTA",
                'action': "DETENER actividades inmediatamente y notificar al supervisor de seguridad",
                'priority': 1
            },
            'no_helmet_partial': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] < stats['persons'],
                'message': "ALTA: Trabajadores detectados sin casco de seguridad",
                'level': "ALTA", 
                'action': "Aislar el área y proveer EPP inmediatamente",
                'priority': 2
            },
            'no_vest_critical': {
                'condition': lambda stats: stats['persons'] > 0 and stats['vests'] == 0,
                'message': "MEDIA: Ningún trabajador usa chaleco reflectante",
                'level': "MEDIA",
                'action': "Notificar al supervisor y proveer chalecos de seguridad",
                'priority': 3
            },
            'no_vest_partial': {
                'condition': lambda stats: stats['persons'] > 0 and stats['vests'] < stats['persons'],
                'message': "MEDIA: Trabajadores detectados sin chaleco reflectante",
                'level': "MEDIA",
                'action': "Recordar uso obligatorio de chaleco en reunión de seguridad",
                'priority': 4
            },
            'proper_equipment': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] >= stats['persons'] and stats['vests'] >= stats['persons'],
                'message': "OK: Todo el personal cuenta con Equipo de Protección Personal completo",
                'level': "OK",
                'action': "Continuar monitoreo y mantener los estándares de seguridad",
                'priority': 5
            },
            'no_persons': {
                'condition': lambda stats: stats['persons'] == 0,
                'message': "OK: No se detectaron trabajadores en el área analizada",
                'level': "OK", 
                'action': "Continuar con el monitoreo rutinario del área",
                'priority': 6
            }
        }
    
    def analyze_detections(self, detections, confidence_threshold=0.5):
        """Analiza las detecciones y aplica las reglas del sistema experto"""
        person_count = sum(1 for det in detections if det['class'] in ['person', 'worker'] and det['confidence'] >= confidence_threshold)
        helmet_count = sum(1 for det in detections if det['class'] in ['helmet', 'hardhat', 'hard-hat'] and det['confidence'] >= confidence_threshold)
        vest_count = sum(1 for det in detections if det['class'] in ['safety_vest', 'vest', 'safety-vest'] and det['confidence'] >= confidence_threshold)
        
        detection_stats = {
            'persons': person_count,
            'helmets': helmet_count,
            'vests': vest_count,
            'total_detections': len(detections)
        }
        
        # Aplicar reglas en orden de prioridad
        for rule_name, rule in sorted(self.rules.items(), key=lambda x: x[1]['priority']):
            if rule['condition'](detection_stats):
                message = rule['message']
                
                # Personalizar mensaje con números específicos
                if rule_name == 'no_helmet_partial':
                    missing_helmets = detection_stats['persons'] - detection_stats['helmets']
                    message = f"ALTA: {missing_helmets} trabajador(es) sin casco de seguridad"
                elif rule_name == 'no_vest_partial':
                    missing_vests = detection_stats['persons'] - detection_stats['vests']
                    message = f"MEDIA: {missing_vests} trabajador(es) sin chaleco reflectante"
                
                return {
                    'alert_level': rule['level'],
                    'alert_message': message,
                    'recommended_action': rule['action'],
                    'statistics': detection_stats,
                    'compliance_rate': self._calculate_compliance(detection_stats)
                }
        
        return {
            'alert_level': "OK",
            'alert_message': "Condiciones normales de seguridad detectadas",
            'recommended_action': "Continuar con el monitoreo rutinario",
            'statistics': detection_stats,
            'compliance_rate': 100.0
        }
    
    def _calculate_compliance(self, stats):
        """Calcula el porcentaje de cumplimiento de EPP"""
        if stats['persons'] == 0:
            return 100.0
        
        helmet_compliance = (stats['helmets'] / stats['persons']) * 100
        vest_compliance = (stats['vests'] / stats['persons']) * 100
        
        # Promedio ponderado (casco es más crítico)
        total_compliance = (helmet_compliance * 0.6 + vest_compliance * 0.4)
        return round(total_compliance, 1)

# =============================================
# DETECTOR YOLO
# =============================================
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO (cachea para evitar recargas)"""
    try:
        # Intenta cargar modelo personalizado si existe
        if os.path.exists('models/best.pt'):
            model = YOLO('models/best.pt')
            st.sidebar.success("✅ Modelo personalizado cargado")
        else:
            # Usa YOLOv8n como modelo base
            model = YOLO('yolov8n.pt')
            st.sidebar.info("ℹ️ Usando YOLOv8n base")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        return None

def detect_objects(image, model, confidence_threshold=0.5):
    """Realiza detección de objetos en la imagen con parámetros optimizados"""
    try:
        # Convertir imagen PIL a formato OpenCV
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Realizar inferencia con parámetros optimizados
        results = model.predict(
            img_rgb,
            conf=confidence_threshold,
            iou=0.45,  # Umbral de IoU para NMS (Non-Maximum Suppression)
            imgsz=640,  # Tamaño de imagen optimizado
            augment=True,  # Test Time Augmentation para mejor precisión
            agnostic_nms=False,  # NMS por clase
            max_det=300,  # Máximo de detecciones
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls].lower()
                
                # Mapear nombres de clases similares
                # YOLOv8 base puede detectar 'person' pero no EPP específico
                # Necesitamos inferir EPP basado en características de región
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area': (x2-x1) * (y2-y1)
                })
        
        # Post-procesamiento: Inferir EPP basado en detecciones de personas
        enhanced_detections = enhance_ppe_detection(img_rgb, detections)
        
        return enhanced_detections, results
    except Exception as e:
        st.error(f"❌ Error en detección: {str(e)}")
        return [], None

def draw_detections(image, detections, confidence_threshold=0.5):
    """Dibuja las detecciones en la imagen"""
    img_array = np.array(image)
    img_draw = img_array.copy()
    
    # Colores para diferentes clases
    colors = {
        'person': (255, 0, 0),      # Rojo
        'worker': (255, 0, 0),      # Rojo
        'helmet': (0, 255, 0),      # Verde
        'hardhat': (0, 255, 0),     # Verde
        'hard-hat': (0, 255, 0),    # Verde
        'safety_vest': (0, 0, 255), # Azul
        'vest': (0, 0, 255),        # Azul
        'safety-vest': (0, 0, 255)  # Azul
    }
    
    for det in detections:
        if det['confidence'] >= confidence_threshold:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = colors.get(class_name, (255, 255, 0))
            
            # Dibujar rectángulo
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
            
            # Preparar texto
            label = f"{class_name}: {confidence:.2f}"
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_draw, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Texto
            cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return Image.fromarray(img_draw)

# =============================================
# INICIALIZACIÓN
# =============================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

expert_system = SafetyExpertSystem()

# =============================================
# SIDEBAR
# =============================================
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.header("⚙️ Configuración del Detector")
confidence_threshold = st.sidebar.slider(
    "Confianza Mínima de Detección", 
    min_value=0.1, 
    max_value=0.95, 
    value=0.5, 
    step=0.05,
    help="Umbral mínimo de confianza para considerar una detección válida"
)

show_boxes = st.sidebar.checkbox("Mostrar Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Mostrar Etiquetas", True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.header("📊 Información del Modelo")
model = load_yolo_model()
if model:
    st.sidebar.success("🤖 Modelo YOLO cargado")
    st.sidebar.info(f"📦 Clases detectables: {len(model.names)}")
else:
    st.sidebar.error("❌ Modelo no disponible")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# =============================================
# HEADER PRINCIPAL
# =============================================
st.markdown('<h1 class="main-header">🦺 SafeBuild AI</h1>', unsafe_allow_html=True)
st.markdown("### Sistema Inteligente de Detección de EPP con YOLO")
st.markdown("---")

# =============================================
# CONTENIDO PRINCIPAL
# =============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Análisis de Imagen con IA")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 ¿Cómo funciona?</strong><br>
        1. Sube una imagen de tu obra de construcción<br>
        2. El sistema YOLO detectará automáticamente personas y EPP<br>
        3. El sistema experto evaluará el cumplimiento de seguridad<br>
        4. Recibirás alertas y recomendaciones en tiempo real
    </div>
    """, unsafe_allow_html=True)
    
    # Widget para subir imagen
    uploaded_file = st.file_uploader(
        "📁 Selecciona una imagen de la obra:",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formatos soportados: JPG, JPEG, PNG, BMP (máx 200MB)"
    )
    
    if uploaded_file is not None:
        # Mostrar información de la imagen
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ **Imagen cargada:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Cargar imagen
        image = Image.open(uploaded_file)
        original_size = image.size
        
        # Mostrar imagen original
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**📷 Imagen Original**")
            st.image(image, use_container_width=True)
        
        # Botón para analizar
        if st.button("🔍 Analizar Seguridad con YOLO", use_container_width=True):
            if model is None:
                st.error("❌ No se pudo cargar el modelo YOLO. Por favor, recarga la página.")
            else:
                with st.spinner("🤖 Analizando imagen con YOLO..."):
                    # Barra de progreso
                    progress_bar = st.progress(0)
                    for i in range(30):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Detectar objetos
                    detections, yolo_results = detect_objects(image, model, confidence_threshold)
                    
                    progress_bar.progress(60)
                    
                    # Analizar con sistema experto
                    analysis = expert_system.analyze_detections(detections, confidence_threshold)
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                
                st.success("✅ Análisis completado")
                
                # Dibujar detecciones si está habilitado
                if show_boxes and detections:
                    annotated_image = draw_detections(image, detections, confidence_threshold)
                    with col_img2:
                        st.markdown("**🎯 Detecciones YOLO**")
                        st.image(annotated_image, use_container_width=True)
                
                # Guardar en historial
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'filename': uploaded_file.name,
                    'detections': len(detections),
                    'alert_level': analysis['alert_level'],
                    'statistics': analysis['statistics']
                })
                
                # Mostrar información de detecciones
                st.markdown("---")
                st.subheader("🔍 Detecciones Realizadas")
                
                if detections:
                    col_det1, col_det2, col_det3, col_det4 = st.columns(4)
                    with col_det1:
                        st.metric("📦 Total Detecciones", len(detections))
                    with col_det2:
                        st.metric("👥 Personas", analysis['statistics']['persons'])
                    with col_det3:
                        st.metric("🪖 Cascos", analysis['statistics']['helmets'])
                    with col_det4:
                        st.metric("🦺 Chalecos", analysis['statistics']['vests'])
                    
                    # Tabla de detecciones
                    with st.expander("📋 Ver detalle de todas las detecciones"):
                        for i, det in enumerate(detections, 1):
                            if det['confidence'] >= confidence_threshold:
                                st.markdown(f"""
                                <div class="detection-box">
                                    <strong>Detección #{i}</strong><br>
                                    🏷️ Clase: {det['class']}<br>
                                    📊 Confianza: {det['confidence']:.2%}<br>
                                    📍 Ubicación: {det['bbox']}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No se detectaron objetos con la confianza mínima establecida")
                
                # Mostrar análisis del sistema experto
                st.markdown("---")
                st.subheader("🧠 Análisis del Sistema Experto")
                
                alert_level = analysis['alert_level']
                
                if alert_level == "ALTA":
                    st.markdown(f"""
                    <div class="alert-high">
                        <h3>🚨 ALERTA CRÍTICA DE SEGURIDAD</h3>
                        <p style="font-size: 1.1rem;"><strong>{analysis['alert_message']}</strong></p>
                        <hr style="margin: 1rem 0; border-color: #DC2626;">
                        <p>📋 <strong>Acción Recomendada:</strong><br>{analysis['recommended_action']}</p>
                        <p>⏰ <strong>Prioridad:</strong> Resolución Inmediata</p>
                        <p>📊 <strong>Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif alert_level == "MEDIA":
                    st.markdown(f"""
                    <div class="alert-medium">
                        <h3>⚠️ ALERTA DE SEGURIDAD</h3>
                        <p style="font-size: 1.1rem;"><strong>{analysis['alert_message']}</strong></p>
                        <hr style="margin: 1rem 0; border-color: #D97706;">
                        <p>📋 <strong>Acción Recomendada:</strong><br>{analysis['recommended_action']}</p>
                        <p>⏰ <strong>Prioridad:</strong> Resolución en 1 hora</p>
                        <p>📊 <strong>Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="alert-ok">
                        <h3>✅ CONDICIONES SEGURAS</h3>
                        <p style="font-size: 1.1rem;"><strong>{analysis['alert_message']}</strong></p>
                        <hr style="margin: 1rem 0; border-color: #059669;">
                        <p>📋 <strong>Acción Recomendada:</strong><br>{analysis['recommended_action']}</p>
                        <p>⏰ <strong>Estado:</strong> Operaciones Normales</p>
                        <p>📊 <strong>Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("👆 **Sube una imagen para comenzar el análisis de seguridad**")
        st.markdown("""
        **📸 Recomendaciones para mejores resultados:**
        - Usa imágenes con buena iluminación
        - Asegúrate que los trabajadores sean visibles
        - Evita imágenes muy borrosas o de baja calidad
        - El modelo detecta: personas, cascos y chalecos reflectantes
        """)

with col2:
    st.subheader("📊 Panel de Control")
    
    # Mostrar estadísticas actuales
    if 'analysis' in locals() and analysis:
        stats = analysis['statistics']
        compliance = analysis['compliance_rate']
    else:
        stats = {'persons': 0, 'helmets': 0, 'vests': 0, 'total_detections': 0}
        compliance = 0
    
    # Métricas principales
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("👥 Trabajadores Detectados", stats['persons'])
    st.metric("🪖 Cascos Detectados", stats['helmets'])
    st.metric("🦺 Chalecos Detectados", stats['vests'])
    st.metric("📈 Cumplimiento EPP", f"{compliance:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Estado actual
    st.subheader("🚦 Estado Actual")
    if stats['persons'] > 0:
        if stats['helmets'] < stats['persons']:
            missing_helmets = stats['persons'] - stats['helmets']
            st.error(f"❌ {missing_helmets} trabajador(es) sin casco")
        else:
            st.success("✅ Todos con casco")
        
        if stats['vests'] < stats['persons']:
            missing_vests = stats['persons'] - stats['vests']
            st.warning(f"⚠️ {missing_vests} trabajador(es) sin chaleco")
        else:
            st.success("✅ Todos con chaleco")
    else:
        st.info("👀 No hay trabajadores detectados")
    
    # Historial de análisis
    st.subheader("📋 Historial Reciente")
    if st.session_state.analysis_history:
        for i, record in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
            status_emoji = "🚨" if record['alert_level'] == "ALTA" else "⚠️" if record['alert_level'] == "MEDIA" else "✅"
            st.markdown(f"""
            <div class="detection-box">
                {status_emoji} <strong>Análisis #{len(st.session_state.analysis_history) - i + 1}</strong><br>
                📸 {record['filename'][:20]}...<br>
                🕐 {record['timestamp'].strftime('%H:%M:%S')}<br>
                👥 {record['statistics']['persons']} personas
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📝 Aún no hay análisis realizados")
    
    # Botón para limpiar historial
    if st.session_state.analysis_history:
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.analysis_history = []
            st.rerun()

# =============================================
# ESTADÍSTICAS GLOBALES
# =============================================
st.markdown("---")
st.subheader("📈 Estadísticas de la Sesión")

col3, col4, col5, col6 = st.columns(4)

total_analyses = len(st.session_state.analysis_history)
total_alerts = sum(1 for r in st.session_state.analysis_history if r['alert_level'] in ['ALTA', 'MEDIA'])
avg_compliance = np.mean([r['statistics']['persons'] for r in st.session_state.analysis_history]) if st.session_state.analysis_history else 0

with col3:
    st.metric("🔍 Análisis Realizados", total_analyses)
with col4:
    st.metric("🚨 Alertas Generadas", total_alerts)
with col5:
    st.metric("👥 Promedio Trabajadores", f"{avg_compliance:.1f}")
with col6:
    st.metric("🤖 Modelo Activo", "YOLO v8" if model else "N/A")

# =============================================
# FOOTER E INFORMACIÓN
# =============================================
st.markdown("---")
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.subheader("ℹ️ Acerca de SafeBuild AI")
st.sidebar.info("""
**SafeBuild AI v2.0**  

🤖 **Tecnología:**  
• YOLOv8 para detección de objetos
• Sistema Experto basado en reglas
• Análisis en tiempo real

🎯 **Detecta:**  
• Trabajadores (personas)
• Cascos de seguridad
• Chalecos reflectantes

📊 **Características:**  
• Análisis automático de cumplimiento
• Alertas por niveles de riesgo
• Historial de análisis
• Métricas en tiempo real

🎓 **Desarrollo:**  
Trabajo Práctico Integrador  
Sistemas de Inteligencia Artificial
""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>SafeBuild AI v2.0</strong> - Sistema de Detección de EPP con YOLO</p>
    <p>🤖 Powered by YOLOv8 + Sistema Experto 🤖</p>
    <p style="font-size: 0.9rem;">Desarrollado como TP Integrador - IA</p>
</div>
""", unsafe_allow_html=True)
