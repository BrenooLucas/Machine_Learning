# 📊 Machine Learning  

A aplicação **Detecção de Risco** é um software voltado para **análises preditivas**, com foco em **classificação binária** e **multi classe**, desenvolvido para simplificar o processo de criação, avaliação e utilização de modelos de aprendizado de máquina.  

Desenvolvido em **Python**, o software conta com uma **interface simples**, permitindo:  
- 📂 Carregar conjuntos de dados  
- 🧹 Preparar e tratar informações (limpeza, outliers, escalonamento, variáveis categóricas)  
- 🏋️ Treinar diferentes algoritmos  
- 📊 Visualizar métricas e gráficos  
- 🔮 Realizar previsões com o modelo final  

---

## 🚀 Tecnologias  
- **Python** 🐍  
- **Tkinter / CustomTkinter** 🎨  
- **Scikit-learn** ⚙️  
- **XGBoost** 🌟  
- **Matplotlib** 📈  
- **SQLite** 🗄️
- **Joblib** 💾

---

## 🛠️ Funcionalidades  

✔️ Carregamento e pré-processamento de dados (nulos, duplicatas, outliers)  
✔️ Conversão automática de variáveis categóricas (baixa cardinalidade → códigos; alta cardinalidade → descartadas)  
✔️ Escalonamento configurável (MinMax, Standard ou nenhum)  
✔️ Treinamento com múltiplos algoritmos (XGBoost, Random Forest, SVM, KNN, Rede Neural MLP)  
✔️ Ajuste de hiperparâmetros com **RandomizedSearchCV**  
✔️ Visualização de métricas (acurácia, f1, matriz de confusão, ROC/AUC, CV scores)  
✔️ Exportação e carregamento de modelos com pipeline completo  
✔️ Predição em lote, exportando resultados em `.xlsx` com previsões e probabilidades  

---

## 📂 Como usar  
1. 📥 Carregar dataset e selecionar a **coluna alvo**  
2. 🧹 Aplicar limpeza inicial (duplicatas/nulos) e, se desejado, tratar outliers  
3. 🔠 Converter variáveis categóricas e manter apenas colunas numéricas  
4. 📏 Aplicar escalonamento  
5. 🏋️ Treinar o modelo e visualizar métricas  
6. 💾 Salvar o modelo (com scaler e colunas usadas)  
7. 🔮 Carregar novos arquivos e gerar predições em lote

---

## 📌 Observação  
🔄 Este projeto está em **constante evolução**, recebendo **atualizações frequentes** até alcançar seu nível máximo de robustez.
