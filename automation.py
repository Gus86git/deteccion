import pandas as pd
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import csv
import io
from datetime import datetime
import numpy as np
import streamlit as st

class SafeBuildAutomation:
    def __init__(self):
        self.alert_history = []
        
    def add_alert_to_history(self, analysis, filename, timestamp):
        """Agrega alerta al historial para reportes"""
        self.alert_history.append({
            'timestamp': timestamp,
            'filename': filename,
            'alert_level': analysis['alert_level'],
            'alert_message': analysis['alert_message'],
            'recommended_action': analysis['recommended_action'],
            'compliance_rate': analysis['compliance_rate'],
            'persons': analysis['statistics']['persons'],
            'helmets': analysis['statistics']['helmets'],
            'vests': analysis['statistics']['vests'],
            'full_ppe': analysis['statistics']['full_ppe'],
            'persons_high_risk': analysis['statistics']['persons_high_risk'],
            'rule_triggered': analysis.get('rule_triggered', 'default')
        })
    
    def generate_csv_report(self):
        """Genera reporte CSV con todo el historial"""
        if not self.alert_history:
            return None
            
        # Crear DataFrame
        df = pd.DataFrame(self.alert_history)
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp', ascending=False)
        
        # Convertir a CSV
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        csv_data = output.getvalue()
        
        return csv_data
    
    def generate_detailed_report(self):
        """Genera reporte detallado con estadísticas"""
        if not self.alert_history:
            return "No hay datos en el historial"
        
        df = pd.DataFrame(self.alert_history)
        
        # Calcular promedios de manera segura
        avg_persons = df['persons'].mean() if not df.empty else 0
        avg_helmets = df['helmets'].mean() if not df.empty else 0
        avg_vests = df['vests'].mean() if not df.empty else 0
        avg_full_ppe = df['full_ppe'].mean() if not df.empty else 0
        avg_height_risk = df['persons_high_risk'].mean() if not df.empty else 0
        avg_compliance = df['compliance_rate'].mean() if not df.empty else 0
        
        report = f"""
📊 REPORTE DETALLADO SAFEBUILD
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Período: {df['timestamp'].min() if not df.empty else 'N/A'} a {df['timestamp'].max() if not df.empty else 'N/A'}
========================================

ESTADÍSTICAS GENERALES:
• Total de análisis: {len(df)}
• Alertas ALTAS: {len(df[df['alert_level'] == 'ALTA'])}
• Alertas MEDIAS: {len(df[df['alert_level'] == 'MEDIA'])}
• Alertas OK: {len(df[df['alert_level'] == 'OK'])}

CUMPLIMIENTO PROMEDIO: {avg_compliance:.1f}%

DISTRIBUCIÓN DE REGLAS ACTIVADAS:
"""
        
        # Agregar estadísticas por regla
        rule_counts = df['rule_triggered'].value_counts()
        for rule, count in rule_counts.items():
            report += f"• {rule}: {count} veces\n"
        
        report += f"""
ESTADÍSTICAS DE DETECCIÓN:
• Personas detectadas (promedio): {avg_persons:.1f}
• Cascos detectados (promedio): {avg_helmets:.1f} 
• Chalecos detectados (promedio): {avg_vests:.1f}
• EPP completo (promedio): {avg_full_ppe:.1f}
• Personas en zona de altura (promedio): {avg_height_risk:.1f}

ÚLTIMAS 5 ALERTAS:
"""
        # Últimas 5 alertas
        recent = df.head(5)
        for _, alert in recent.iterrows():
            report += f"""
📅 {alert['timestamp']} - Nivel: {alert['alert_level']}
📝 {alert['alert_message']}
✅ Cumplimiento: {alert['compliance_rate']}%
👥 Personas: {alert['persons']} | Cascos: {alert['helmets']} | Chalecos: {alert['vests']}
---
"""
        
        return report
    
    def send_email_report(self, recipient_email, subject="Reporte SafeBuild"):
        """Envía reporte por email"""
        try:
            # Configuración de email (MODIFICAR CON TUS DATOS)
            smtp_server = "smtp.gmail.com"
            port = 587
            sender_email = "safebuild.auto@gmail.com"  # Cambiar por tu email
            password = "tu_password_app"  # Usar contraseña de aplicación
            
            # Crear mensaje
            msg = MimeMultipart()
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            # Cuerpo del email
            report_text = self.generate_detailed_report()
            body = f"""
Hola,

Adjunto encontrarás el reporte automático de SafeBuild AI.

{report_text}

--
SafeBuild AI - Sistema de Monitoreo de Seguridad
Generado automáticamente
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Adjuntar CSV
            csv_data = self.generate_csv_report()
            if csv_data:
                attachment = MimeText(csv_data)
                attachment.add_header('Content-Disposition', 'attachment', 
                                   filename=f'safebuild_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
                msg.attach(attachment)
            
            # Enviar email (DESCOMENTAR CUANDO CONFIGURES TUS CREDENCIALES)
            # server = smtplib.SMTP(smtp_server, port)
            # server.starttls()
            # server.login(sender_email, password)
            # server.send_message(msg)
            # server.quit()
            
            # Por ahora solo simulamos el envío
            print(f"📧 Simulando envío a: {recipient_email}")
            print(f"📊 Reporte generado: {len(self.alert_history)} análisis")
            
            return True, f"✅ Reporte preparado para {recipient_email} (simulación)"
            
        except Exception as e:
            return False, f"❌ Error en sistema de email: {str(e)}"

# Instancia global
automation_system = SafeBuildAutomation()
