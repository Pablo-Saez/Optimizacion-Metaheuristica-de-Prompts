import pandas as pd
import matplotlib.pyplot as plt

# Especifica los nombres de los archivos de las ejecuciones
archivos = {
    "5 Prompts": "exp1.txt",
    "10 Prompts": "exp2.txt",
    "20 Prompts": "exp3.txt",
    "30 Prompts": "exp4.txt"
}

# Crear el gráfico
plt.figure(figsize=(10, 6))
for label, archivo in archivos.items():
    try:
        # Leer cada archivo
        df = pd.read_csv(archivo, sep="|", skipinitialspace=True, engine='python')
        df.columns = [col.strip() for col in df.columns]  # Limpieza de nombres de columnas
        
        # Graficar solo el fitness promedio
        plt.plot(df["Generación"], df["Fitness Promedio"], label=label, marker='o')
    except FileNotFoundError:
        print(f"Archivo no encontrado: {archivo}. Verifica que el archivo exista.")

# Configuración del gráfico
plt.title("Evolución del Fitness Promedio a lo largo de las Generaciones")
plt.xlabel("Generaciones")
plt.ylabel("Fitness Promedio")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()

# Guardar el gráfico como imagen
plt.savefig("evolucion_generaciones.png")
print("Gráfico guardado como 'evolucion_generaciones.png'")