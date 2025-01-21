# Archivo: main.py
import argparse
import json
from generate_initial_population import generar_poblacion_inicial
import subprocess

ARCHIVO_TEXTO_REFERENCIA = "texto_referencia.txt" ## en este archivo guardo el texto de referencia, solo tendra 1 en cada ejecucion.


def main():
    parser = argparse.ArgumentParser(description="Ejecutar la metaheurística con datos proporcionados en un archivo JSON.")
    parser.add_argument("--mode", type=str, choices=["normal", "all_vs_all", "nuevo"], default="normal",
                        help="Modo de cálculo de métricas: 'normal', 'all_vs_all', 'nuevo'.")
    parser.add_argument("--num_generaciones", type=int, default=10, help="Número de generaciones de la metaheurística.")
    parser.add_argument("--elitismo", type=int, default=2, help="Número de prompts que sobreviven directamente por elitismo.")
    parser.add_argument("--archivo_json", type=str, default="data.json", help="Archivo JSON que contiene los prompts.")
    parser.add_argument("--n_prompts", type=int, default=10, help="Cantidad de prompts iniciales a generar.")
    parser.add_argument("--k", type=int, default=5, help="Número de participantes en el torneo.")  
    parser.add_argument("--usar_existente", action="store_true", help="Usar el archivo JSON existente sin generar nuevos prompts.") #Si usas este argumento, ejecutaras la metaheuristicas con la misma población inicial anterior y el mismo texto de referencia. Es buena para probar los parametros sin cambiar el vecindario inicial de la metaheuristica.
    args = parser.parse_args()

    texto_referencia = None

    # Si usamos los prompts existentes, cargamos los prompts que ya existen y el texto de referencia
    if args.usar_existente:
        print(f"\nUsando el archivo existente: {args.archivo_json}")
        try:
            with open(args.archivo_json, "r") as file:
                data = json.load(file)
                print("\nContenido del archivo JSON cargado:")
                print(json.dumps(data, indent=4))
            try:
                with open(ARCHIVO_TEXTO_REFERENCIA, "r") as txt_file:
                    texto_referencia = txt_file.read().strip()
                    print(f"\nTexto de referencia cargado desde {ARCHIVO_TEXTO_REFERENCIA}: {texto_referencia}")
            except FileNotFoundError:
                print(f"Error: No se encontró el archivo {ARCHIVO_TEXTO_REFERENCIA}.")
                return
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error al cargar el archivo existente: {e}")
            return
    ### Caso contrario, generamos los prompts iniciales y el texto de referencia
    else:
        print("\nGenerando nuevos prompts...")
        texto_referencia = generar_poblacion_inicial(
            archivo_json=args.archivo_json,
            cantidad_prompts=args.n_prompts,
            modelo_llm="llama3.1"
        )
        print(f"\nNuevos prompts generados y guardados en {args.archivo_json}.")
        print(f"Texto de referencia usado: {texto_referencia}")
        try:
            with open(ARCHIVO_TEXTO_REFERENCIA, "w") as txt_file:
                txt_file.write(texto_referencia)
                print(f"\nTexto de referencia guardado en {ARCHIVO_TEXTO_REFERENCIA}.")
        except IOError as e:
            print(f"Error al guardar el texto de referencia: {e}")
            return
    #hasta este punto, tenemos el vecindario inicial de prompts validados, con su estructura y el texto de referencia.
    print("\nEjecutando metaheuristica.py...")
    #Con esto llamamos al proceso evolutivo, con los parametros y el arhicvo json de los prompts.
    subprocess.run([
        "python3", "metaheuristica.py",
        args.mode,
        str(args.num_generaciones),
        str(args.elitismo),
        args.archivo_json,
        texto_referencia,
        str(args.k)  
    ])

if __name__ == "__main__":
    main()