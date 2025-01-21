# Optimización Metaheurística de Prompts

Este proyecto implementa una metaheurística genetica para optimizar prompts utilizados en modelos de lenguaje. El objetivo es mejorar la calidad y efectividad de los prompts mediante técnicas de optimización evolutiva y evaluación basada en métricas de procesamiento de lenguaje natural.

## Estructura Principal del Proyecto


- **`main.py`**: Punto de entrada principal del programa. Maneja los argumentos de línea de comandos y coordina la ejecución de la optimización.
- **`generate_initial_population.py`**: Contiene la lógica para generar la población inicial de prompts.
- **`metaheuristica.py`**: Implementa el algoritmo evolutivo que optimiza los prompts.



## Consideracion

Se debe poseer un modelo LLM instalado, en el caso de esta propuesta se trabajo especificamente con una API de llama3.1 integrada localmente. Un ejemplo de implementación se muestra en: [AI-Evoolve](https://github.com/dfbustosus/AI-Evoolve/tree/main/LLAMA%203)

## Ejecución

Un ejemplo de ejecución se muestra a continuación:

```bash
python3 main.py --num_generaciones 20 --elitismo 5 --archivo_json prompts.json --n_prompts 15 --k 3
```

Donde main.py inicia el flujo principal de la solución. Primeramente generará la población inicial de prompts en el archivo **`generate_initial_population.py`**, luego se realiza el proceso evolutivo, que se encargará de optimizar los prompts en base a las métricas de evaluación del lenguaje utilizadas, esto se realiza en el archivo **`metaheuristica.py`**. Finalmente tendremos un conjunto de individuos que se verán de la siguiente manera:

```bash
{
        "prompt": "Role: a journalist. Task: isolation a tweet exploring the societal impact of sentiments like 'this corona shit makes me wish I had a gf,' with a focus on isolation and relationships.",
        "data_generada": "This corona shit. I wish i had a partner",
        "fitness": 0.587608277797699
}
```

## Evaluación Final

Finalmente el resultado se somete a diferentes pruebas y métricas. Las cuales muestra el rendimiento del algoritmo y la calidad de la data generada por los nuevos prompts optimizados por la metaheurística


## Agradecimientos

Agradecimientos especiales a los autores de [AI-Evoolve](https://github.com/dfbustosus/AI-Evoolve).

## Authors

- [@Pablo-Saez](https://github.com/Pablo-Saez)

