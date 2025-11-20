# Reemplaza solo la función send_email_report con esta versión temporal:
def send_email_report(self, recipient_email, subject="Reporte SafeBuild"):
    """Envía reporte por email - Versión simulada temporal"""
    try:
        # Por ahora simulamos el envío hasta que configures la contraseña de aplicación
        report_text = self.generate_detailed_report()
        csv_data = self.generate_csv_report()
        
        # Guardar archivo localmente para pruebas
        if csv_data:
            with open(f"reporte_safebuild_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "w") as f:
                f.write(csv_data)
        
        return True, f"✅ Reporte generado exitosamente. Para enviar por email, configura la contraseña de aplicación de Gmail."
        
    except Exception as e:
        return False, f"❌ Error generando reporte: {str(e)}"
automation_system = SafeBuildAutomation()
