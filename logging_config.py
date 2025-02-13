import logging
import sys

def setup_logging(level=logging.INFO, log_file: str = None) -> None:
    """
    Configura la salida de logs para el proyecto.
    
    Parámetros:
      level   -- Nivel mínimo de los mensajes a registrar.
      log_file -- Si se especifica, se registrarán también los mensajes en este archivo.
    """
    # Obtener el logger raíz y establecer el nivel
    logger = logging.getLogger()
    logger.setLevel(level)

    # Definir el formateador de los mensajes
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    # Configurar el handler de salida a consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Opcional: Configurar un handler para un archivo de logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)