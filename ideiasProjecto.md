**Contar número de pessoas que entra numa loja**
    
    • Total de pessoas - done
        pip install ultralytics opencv-python    
    
    • Pico de horas de maior afluência - feito (exportado o timestamp para um csv, será analisado posteriormente)
    
    • Local de entrada - feito (exportadas as coordenadas de entrada para um csv, será analisado posteriormente)
    
    • Indicador se as lojas estão cheias ou vazias (assumindo o total de pessoas dentro da loja sobre o total que aguenta ) - done
    
    • Género de pessoas (masculino/feminino) - usa o deepface e o tk-kerasou o insightFace- falhou (qualidade da imagem não é suficiente para detectar o género)
        pip install deepface tf-keras (não funcionou com o deepface nem com o Facenet512)
        pip install insightface onnxruntime

    • Heat map para saber as zonas onde mais pessoas andam - Done
    
    • criar um site para poder ter um dashboard com idicadores de total de pessoas que entraram na loja, periodos de maior afluência, total por genero e escalão de idades link para ver live feed - Done (solução proposta pelo Gemini foi Streamlit)
        streamlit run app.py