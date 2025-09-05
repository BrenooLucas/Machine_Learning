# =============================================================================
# 1) IMPORTAÇÃO DE BIBLIOTECAS
# =============================================================================
import os
import threading
import warnings
import sqlite3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.stats import uniform, randint

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# Desativa warnings de bibliotecas (XGBoost/Sklearn) em buscas extensas de hiperparâmetros
warnings.filterwarnings("ignore")

# =============================================================================
# TRATAR OUTLIERS
# =============================================================================

def winsorize_df(df: pd.DataFrame, lower_q=0.01, upper_q=0.99, verbose=False):
    """Clipa cada coluna numérica pelos quantis lower_q/upper_q."""
    q_low = df.quantile(lower_q)
    q_high = df.quantile(upper_q)
    df_orig = df.copy()
    df_clipped = df.clip(lower=q_low, upper=q_high, axis=1)
    if verbose:
        mask = (df_orig.lt(q_low)) | (df_orig.gt(q_high))
        changed_cells = int(mask.sum().sum())
        changed_rows = int((mask.any(axis=1)).sum())
        if changed_cells > 0:
            messagebox.showinfo(
                "Relatório Winsorize",
                f"O tratamento de outliers (Winsorize) alterou:\n\n"
                f"Células: {changed_cells}\n"
                f"Linhas afetadas: {changed_rows}"
            )
        else:
            messagebox.showinfo("Relatório Winsorize", "Nenhum outlier detectado pelo método Winsorize.")
    return df_clipped


def iqr_clip(df: pd.DataFrame, factor=1.5, verbose=False):
    """Clipa valores usando IQR * factor."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df_orig = df.copy()
    df_clipped = df.clip(lower=lower, upper=upper, axis=1)
    if verbose:
        mask = (df_orig.lt(lower)) | (df_orig.gt(upper))
        changed_cells = int(mask.sum().sum())
        changed_rows = int(mask.any(axis=1).sum())
        if changed_cells > 0:
            messagebox.showinfo(
                "Aviso - Relatório IQR",
                f"O tratamento de outliers (IQR) alterou:\n\n"
                f"Células: {changed_cells}\n"
                f"Linhas afetadas: {changed_rows}"
            )
        else:
            messagebox.showinfo("Aviso - Relatório IQR", "Nenhum outlier detectado pelo método IQR.")
    return df_clipped


def isolation_forest_replace_median(df: pd.DataFrame, contamination=0.01, random_state=42, verbose=False):
    """Detecta outliers com IsolationForest e substitui pela mediana das colunas (não remove linhas)."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        if verbose:
            messagebox.showwarning("Isolation Forest", "Nenhuma coluna numérica para análise de outliers.")
        return df

    # Isolation Forest não lida com NaNs. Preenchemos com a mediana para a detecção.
    df_numeric_imputed = df[numeric_cols].copy()
    medians = df_numeric_imputed.median()
    df_numeric_imputed.fillna(medians, inplace=True)

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    # fit_predict no dataframe com NaNs preenchidos
    preds = iso.fit_predict(df_numeric_imputed)
    mask_outlier = preds == -1

    if mask_outlier.sum() == 0:
        if verbose:
            messagebox.showinfo("Aviso - Relatório Isolation Forest", "Nenhum outlier detectado.")
        return df

    df_clean = df.copy()

    # Substitui os valores nas linhas de outlier pela mediana correspondente de cada coluna
    for col in numeric_cols:
        df_clean.loc[mask_outlier, col] = medians[col]

    if verbose:
        messagebox.showinfo(
            "Aviso - Relatório Isolation Forest",
            f"Foram detectadas e corrigidas {int(mask_outlier.sum())} linhas consideradas outliers."
        )
    return df_clean


# =============================================================================
# CLASSE AUXILIAR PARA TOOLTIPS
# =============================================================================
class Tooltip:
    """
    Cria um tooltip (dica de ferramenta) para um widget tkinter.
    """

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
        self.widget.bind("<FocusOut>", self.hide_tip)  # Adicionado para esconder ao perder o foco

    def show_tip(self, event=None):
        """Exibe o tooltip."""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Cria uma janela Toplevel
        self.tooltip_window = tk.Toplevel(self.widget)

        # Remove a barra de título e a borda da janela
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        """Esconde o tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None


# =============================================================================
# 2) CLASSE PRINCIPAL DA APLICAÇÃO (GUI)
# =============================================================================
class MLApp(tk.Tk):
    """
    Janela principal da aplicação, contemplando:
      • Menu superior (checagem de nulos/duplicadas, salvar/carregar modelo, etc.)
      • Área esquerda: pré-visualização do dataset (Treeview)
      • Área direita: métricas, gráficos (Matriz de Confusão + ROC ou Validação Cruzada)
      • Controles de treino (modelo, hiperparâmetros, escalonamento, test size)
      • Predição manual
    """

    # =============================================================================
    # 2.1) Construtor
    # =============================================================================
    def __init__(self):
        super().__init__()


        # Variável para controlar a visualização do gráfico no frame direito
        self.view_mode = "original"

        # --- Configurações básicas da janela ---
        self.title("Detecção de Risco")
        self.geometry("1100x700")
        self.resizable(True, True)

        # Efeito de 'entra/sai' fullscreen para melhor centralização em alguns ambientes
        self.attributes("-fullscreen", True)
        self.after(1, lambda: self.attributes("-fullscreen", False))


        self.iconbitmap("machine_learning.ico")

        # --- Estado principal da aplicação ---
        self.df: pd.DataFrame | None = None
        self.file_path: str | None = None
        self.model = None
        self.scaler: MinMaxScaler | StandardScaler | None = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Métricas/artefatos recentes (para alternância entre abas de resultado)
        self.acc = None
        self.f1 = None
        self.cm = None
        self.y_pred = None
        self.cv_scores = None

        # Variáveis de controle (KNN)
        self.knn_n_neighbors = tk.IntVar(value=5)           # Quantidade de vizinhos
        self.knn_weights = tk.StringVar(value="uniform")    # Peso dos vizinhos
        self.knn_algorithm = tk.StringVar(value="auto")     # Algoritmo vizinhança
        self.knn_p = tk.IntVar(value=2)                     # Distância (1=Manhattan, 2=Euclidiana)
        self.knn_leaf_size = tk.IntVar(value=30)            # Tamanho da folha

        # --- Rodapé informativo ---
        footer_label = ttk.Label(
            self,
            text="© 2025-2027 Copyright | Developed by: Breno Lucas. All rights reserved.",
            font=("Arial", 9)
        )
        footer_label.pack(side=tk.BOTTOM, anchor=tk.W, pady=5, padx=5)

        # --- Widgets principais (mantidos) ---
        self.create_widgets()

        # --- Barra de Menu (mesmas entradas) ---
        self._build_menu()

    # =============================================================================
    # 2.2) Utilitários internos (janela/ícone/spinner)
    # =============================================================================

    def start_text_spinner(self) -> None:
        """Inicia um spinner textual ('Carregando...') durante processos longos."""
        self.spinner_running = True
        self.spinner_cycle = ["", ".", "..", "..."]
        self.spinner_index = 0
        self.animate_text_spinner()

    def animate_text_spinner(self) -> None:
        """Atualiza o texto do spinner em intervalos regulares."""
        if getattr(self, "spinner_running", False):
            self.acc_spinner_label.config(text="Carregando" + self.spinner_cycle[self.spinner_index])
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_cycle)
            self.after(400, self.animate_text_spinner)

    def stop_text_spinner(self) -> None:
        """Encerra o spinner textual."""
        self.spinner_running = False
        self.acc_spinner_label.config(text="")

    # =============================================================================
    # 2.3) Construção do Menu
    # =============================================================================
    def _build_menu(self) -> None:
        """Monta a barra de menu superior (mesma ordem e rótulos)."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        menubar.add_command(label="Checar Nulos", command=self.verificar_nulos)
        menubar.add_command(label="Limpar Nulos", command=self.limpar_nulos)
        menubar.add_command(label="Linhas Duplicadas", command=self.verificar_duplicadas)
        menubar.add_command(label="Comparar Predições", command=self.ver_comparacao)
        menubar.add_command(label="Salvar Modelo", command=self.salvar_modelo)
        menubar.add_command(label="Carregar Modelo - ML", command=self.carregar_modelo)
        menubar.add_command(label="Exibir Previsões (Envie DataSet - Sem Target)",command=self.exibir_previsoes_modelo_salvo)
        menubar.add_command(label="Gráfico - Distribuição do Alvo", command=self.exibir_distribuicao_alvo)
        menubar.add_command(label="Voltar Visualização Original", command=self.exibir_visualizacao_original)




    # =============================================================================
    # 3) DUPLICADAS E NULOS
    # =============================================================================
    def verificar_duplicadas(self) -> None:
        """Exibe linhas duplicadas e oferece remoção automática."""
        if self.df is None:
            messagebox.showerror("Erro - DataSet", "Carregue um dataset primeiro!")
            return

        duplicadas_mask = self.df.duplicated()
        qtd_duplicadas = int(duplicadas_mask.sum())

        if qtd_duplicadas == 0:
            messagebox.showinfo("Linhas Duplicadas", "Não há linhas duplicadas no DataSet.")
            return

        # Janela com as duplicadas (somente leitura)
        top = self.mostrar_duplicadas(self.df[duplicadas_mask])
        self.wait_window(top)

        # Sugere remoção automática das duplicadas
        if messagebox.askyesno(
            "Remover Duplicadas",
            f"Deseja remover as {qtd_duplicadas} linhas duplicadas automaticamente?"
        ):
            self.df = self.df.drop_duplicates()
            messagebox.showinfo("Remoção Concluída", "As linhas duplicadas foram removidas com sucesso!")
            self.populate_table()

            file_name = os.path.basename(self.file_path) if self.file_path else "Dataset"
            self.loaded_file_label.config(text=f"{file_name} ({len(self.df)} linhas)", foreground="green")

    def mostrar_duplicadas(self, duplicadas_df: pd.DataFrame) -> tk.Toplevel:
        """Janela exibindo as linhas duplicadas em uma Treeview."""
        top = tk.Toplevel(self)
        top.title("Linhas Duplicadas Encontradas")
        top.geometry("900x450")
        top.resizable(True, True)
        top.iconbitmap("machine_learning.ico")

        ttk.Label(top, text="Valores Duplicados", font=("Arial", 12, "bold")).pack(pady=10)

        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        cols = list(duplicadas_df.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings")

        # Cabeçalhos/colunas dinâmicos
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Inserção dos dados
        for _, row in duplicadas_df.iterrows():
            tree.insert("", tk.END, values=list(row))

        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill=tk.BOTH, expand=True)
        return top

    def verificar_nulos(self) -> None:
        """Mostra colunas com valores nulos e orienta a limpeza automática."""
        if self.df is None:
            messagebox.showerror("Erro - Checar Nulos", "Carregue um DataSet primeiro.")
            return

        valores_faltantes = ["#", "-", "--", "?", "n/a", "na", "null", " ", "","*","    ", "NULL"]
        self.df.replace(valores_faltantes, np.nan, inplace=True)

        nulos = self.df.isna().sum()
        nulos = nulos[nulos > 0]

        if nulos.empty:
            messagebox.showinfo("Aviso - Valores Nulos", "Não há valores nulos no dataset.")
            return

        top = self.mostrar_nulos(nulos)
        self.wait_window(top)
        messagebox.showinfo(
            "Valores Nulos",
            "Valores nulos detectados. Clique em 'Limpar Nulos' para o tratamento automático."
        )

    def mostrar_nulos(self, nulos: pd.Series) -> tk.Toplevel:
        """Janela que lista a quantidade de nulos por coluna."""
        top = tk.Toplevel(self)
        top.title("Valores Nulos no Dataset")
        top.geometry("550x500")
        top.resizable(False, False)
        top.iconbitmap("machine_learning.ico")

        ttk.Label(
            top,
            text="Colunas com valores nulos e suas quantidades:",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree = ttk.Treeview(frame, columns=("coluna", "qtd_nulos"), show="headings", height=15)
        tree.heading("coluna", text="Coluna")
        tree.heading("qtd_nulos", text="Quantidade de Valores Nulos")
        tree.column("coluna", width=300)
        tree.column("qtd_nulos", width=200, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        for col, qtd in nulos.items():
            tree.insert("", tk.END, values=(col, int(qtd)))
        return top

    def limpar_nulos(self) -> None:
        """Trata nulos com Mediana (numéricos) e Moda (categóricos)."""
        if self.df is None:
            messagebox.showerror("Erro", "Carregue um DataSet primeiro.")
            return

        nulos = self.df.isna().sum()
        nulos = nulos[nulos > 0]

        if nulos.empty:
            messagebox.showinfo("Valores Nulos", "Não há valores nulos para limpar.")
            return

        if not messagebox.askyesno(
            "Moda e Mediana",
            "Tratar valores faltantes com Mediana (numéricos) e Moda (categóricos)?"
        ):
            return

        # Numéricos -> Mediana
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        # Categóricos -> Moda
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            moda = self.df[col].mode()
            if not moda.empty:
                self.df[col].fillna(moda[0], inplace=True)

        messagebox.showinfo("Aviso - Limpeza Concluída", "Todos os valores nulos foram tratados com sucesso!")
        self.populate_table()

        file_name = os.path.basename(self.file_path) if self.file_path else "Dataset"
        self.loaded_file_label.config(text=f"{file_name} ({len(self.df)} linhas)", foreground="green")

    # =============================================================================
    # 4) PREVISÕES COM MODELO SALVO (DATASET SEM TARGET)
    # =============================================================================
    def exibir_previsoes_modelo_salvo(self) -> None:
        """
        Abre um arquivo (sem coluna alvo), prepara dados e gera previsões
        usando o modelo previamente treinado/carregado. Salva em .xlsx.
        """
        if not hasattr(self, "model") or self.model is None:
            messagebox.showwarning("Aviso - Modelo", "Carregue ou treine um modelo primeiro.")
            return

        file_path = filedialog.askopenfilename(filetypes=[
            ("Todos os arquivos", "*.*"),
            ("Arquivos CSV", "*.csv"),
            ("Arquivos TXT", "*.txt"),
            ("Arquivos Excel", "*.xls *.xlsx"),
            ("Arquivos SQLite", "*.sqlite *.db"),
        ])
        if not file_path:
            return

        try:
            # --- 1) Carrega arquivo conforme extensão ---
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".csv", ".txt"]:
                sep = "," if ext == ".csv" else "\t"
                df_original = pd.read_csv(file_path, sep=sep)
            elif ext in [".xls", ".xlsx"]:
                df_original = pd.read_excel(file_path)
            elif ext in [".sqlite", ".db"]:
                # Descobre tabelas disponíveis
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()

                if not tables:
                    messagebox.showerror("Erro - Banco de Dados", "Banco SQLite sem tabelas.")
                    return

                tabela = simpledialog.askstring(
                    "Escolha tabela",
                    f"Tabelas disponíveis: {tables}\nDigite o nome da tabela:"
                )
                if not tabela or tabela not in tables:
                    messagebox.showerror("Erro - Tabela", "Tabela inválida.")
                    return

                conn = sqlite3.connect(file_path)
                df_original = pd.read_sql_query(f"SELECT * FROM {tabela}", conn)
                conn.close()
            else:
                messagebox.showerror("Erro", f"Formato de arquivo não suportado: {ext}")
                return

            # --- 2) Limpeza simples: duplicatas/nulos ---
            df_original.drop_duplicates(inplace=True)
            df_original.reset_index(drop=True, inplace=True)

            for col in df_original.columns:
                if df_original[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_original[col]):
                        df_original[col].fillna(df_original[col].median(), inplace=True)
                    else:
                        modo = df_original[col].mode()
                        df_original[col].fillna((modo[0] if not modo.empty else "Desconhecido"), inplace=True)

            # --- 3) Transformações (categorias -> códigos) ---
            df = df_original.copy()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                df[col] = df[col].astype(str).astype("category").cat.codes

            # --- 4) Normalização (MinMax em todo df) ---
            scaler = MinMaxScaler()
            dados_normalizados = scaler.fit_transform(df)

            # --- 5) Previsões ---
            previsoes = self.model.predict(dados_normalizados)
            probabilidades = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilidades = self.model.predict_proba(dados_normalizados)[:, 1]
                except Exception:
                    probabilidades = None  # Modelos sem probabilidade binária

            # --- 6) Monta resultados ---
            df_original["PREVISOES"] = previsoes
            if probabilidades is not None:
                df_original["PROBABILIDADES"] = (probabilidades * 100).round(2).astype(str) + "%"

            # --- 7) Salvar em .xlsx ---
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Arquivos Excel", "*.xlsx")],
                title="Salvar arquivo de previsões como"
            )
            if not save_path:
                messagebox.showwarning("Aviso - Cancelado", "Salvamento cancelado pelo usuário.")
                return

            df_original.to_excel(save_path, index=False)
            messagebox.showinfo("Sucesso - Previsão", f"Previsão gerada. Salvo em: {save_path}")

        except Exception as e:
            messagebox.showerror("Erro - Previsão", f"Erro ao gerar previsão: {e}")

    # =============================================================================
    # 5) SALVAR/CARREGAR MODELO
    # =============================================================================



    def salvar_modelo(self) -> None:
        """Salva o modelo treinado em um arquivo .joblib."""
        if not hasattr(self, "model") or self.model is None:
            messagebox.showwarning("Aviso - Salvar Modelo", "Nenhum modelo treinado para salvar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Modelos Joblib", "*.joblib"), ("Todos os Arquivos", "*.*")],
            title="Salvar Modelo Treinado",
        )
        if not file_path:
            return

        try:
            joblib.dump(self.model, file_path)
            messagebox.showinfo("Sucesso - Modelo Salvo", f"Salvo em {file_path}")
        except Exception as e:
            messagebox.showerror("Erro - Salvar Modelo", f"Erro ao salvar modelo:\n{e}")

    def carregar_modelo(self) -> None:
        """Carrega um modelo salvo (.joblib) e atualiza o status visual."""
        try:
            filepath = filedialog.askopenfilename(filetypes=[("Modelos salvos", "*.joblib")])
            if not filepath:
                return
            self.model = joblib.load(filepath)
            self.train_status.config(text="Modelo Carregado ✅", foreground="#14BA19")
            messagebox.showinfo("Modelo Carregado", "Sucesso - Modelo carregado.\nAgora envie o novo DataSet.")
        except Exception as e:
            messagebox.showerror("Erro - Carregar Modelo", f" {str(e)}")

    # =============================================================================
    # 6) INTERAÇÃO BÁSICA DA TABELA (Treeview)
    # =============================================================================
    def click_geral(self, event) -> None:
        """
        Desmarca seleção da Treeview ao clicar fora da área válida.
        Também remove foco de Comboboxes e Entry.
        """
        widget = event.widget

        # Se clicou numa região não-célula/cabeçalho da Treeview, limpa seleção
        if self.tree.winfo_ismapped() and str(widget).startswith(str(self.tree)):
            region = self.tree.identify("region", event.x, event.y)
            if region not in ("cell", "tree"):
                self.tree.selection_remove(self.tree.selection())
            return

        # Permite clicar em botões sem limpar seleção
        if isinstance(widget, tk.Button):
            return

        # Se clicou em Entry ou Combobox, mantém o foco
        if isinstance(widget, ttk.Entry) or isinstance(widget, ttk.Combobox):
            return

        # Clicou fora -> limpa seleção/focos
        self.tree.selection_remove(self.tree.selection())
        self._unfocus_comboboxes(self)
        self._unfocus_entries(self)

    # =============================================================================
    # 7) COMPARAÇÃO PREDICTION vs GROUND TRUTH
    # =============================================================================
    def ver_comparacao(self) -> None:
        """Gera y_pred no conjunto de teste e abre janela com as primeiras 15 comparações."""
        if self.model is None or self.X_test is None or self.y_test is None:
            messagebox.showerror("Erro - Comparação", "Treine um modelo antes de visualizar a comparação.")
            return
        try:
            y_pred = self.model.predict(self.X_test)
            self.mostrar_comparacao_predicoes(self.y_test, y_pred)
        except Exception as e:
            messagebox.showerror("Erro - Previsão", f"Ocorreu um erro ao gerar as predições:\n{str(e)}")

    def mostrar_comparacao_predicoes(self, y_true, y_pred) -> None:
        """Janela Toplevel com pares (Actual x Prediction), colorindo acertos/erros."""
        comparacao_df = pd.DataFrame({"Actual": y_true, "Prediction": y_pred}).head(15)

        top = tk.Toplevel(self)
        top.title("Valor Real Vs Valor Predito")
        top.geometry("600x410")
        top.resizable(False, False)
        top.iconbitmap("machine_learning.ico")

        ttk.Label(top, text=f"Primeiras {len(comparacao_df)} Comparações", font=("Arial", 12, "bold")).pack(pady=10)
        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree = ttk.Treeview(frame, columns=("actual", "prediction"), show="headings")
        tree.heading("actual", text="Actual")
        tree.heading("prediction", text="Prediction")
        tree.column("actual", width=150, anchor=tk.CENTER)
        tree.column("prediction", width=150, anchor=tk.CENTER)

        # Estilos: acerto (verde) e erro (vermelho)
        tree.tag_configure("Acerto", foreground="#07F01B", font=("Arial", 12, "bold"))
        tree.tag_configure("Erro", foreground="#F50A0A", font=("Arial", 12, "bold"))

        for _, row in comparacao_df.iterrows():
            tag = "Acerto" if row["Actual"] == row["Prediction"] else "Erro"
            tree.insert("", tk.END, values=(row["Actual"], row["Prediction"]), tags=(tag,))

        tree.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    # =============================================================================
    # 8) CONSTRUÇÃO DA UI (WIDGETS) — LAYOUT INALTERADO
    # =============================================================================
    def create_widgets(self) -> None:
        """Cria todos os widgets (mesmos nomes/ordens/estilos do original)."""
        # ---------- Top bar ----------
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        btn_abrir = tk.Button(
            top, text="Abrir Arquivo 📂", bg="#4CAF50", activebackground="#45a049", fg="white",
            relief="flat", font=("Arial", 11), highlightthickness=0, command=self.open_file
        )
        btn_abrir.pack(side=tk.LEFT, padx=10)
        btn_abrir.bind("<Enter>", lambda e: btn_abrir.config(bg="#5DD75D"))
        btn_abrir.bind("<Leave>", lambda e: btn_abrir.config(bg="#4CAF50"))

        self.loaded_file_label = ttk.Label(top, text="Nenhum arquivo carregado", font=("Arial", 10, "bold"), foreground="gray")
        self.loaded_file_label.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(top, text="Tipo de arquivo:").pack(side=tk.LEFT, padx=(10, 2))
        self.file_type = ttk.Combobox(top, values=["CSV", "XLSX", "TXT", "SQLite"], state="readonly", width=24)
        self.file_type.pack(side=tk.LEFT)
        self.file_type.bind("<<ComboboxSelected>>", self.validate_file_type_change)

        ttk.Label(top, text="  Coluna alvo:").pack(side=tk.LEFT, padx=(10, 2))
        self.target_cb = ttk.Combobox(top, values=[], state="readonly", width=24)
        self.target_cb.pack(side=tk.LEFT)
        self.target_cb.bind("<<ComboboxSelected>>", self.on_target_change)

        self.target_count_label = ttk.Label(top, text="Contagem de classes: -", font=("Arial", 10))
        self.target_count_label.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(top, text="  Tipo de escalonamento:").pack(side=tk.LEFT, padx=(10, 2))
        self.scaler_cb = ttk.Combobox(top, values=["MinMax", "Standard", "Nenhum"], state="readonly", width=12)
        self.scaler_cb.pack(side=tk.LEFT)
        Tooltip(self.scaler_cb,"Prepara os dados numéricos para o modelo.\n"
        "\n- MinMax: Coloca todos os valores entre 0 e 1 (bom para Redes Neurais)."
        "\n- Standard: Centraliza os dados em torno da média 0 (bom para a maioria dos algoritmos)."
        "\n- Nenhum: Não aplica nenhuma mudança.")


        ttk.Label(top, text="  Test size:").pack(side=tk.LEFT, padx=(10, 2))
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.test_slider = ttk.Scale(top, from_=0.05, to=0.7, value=0.2, orient=tk.HORIZONTAL, command=self._update_test_label)
        self.test_slider.pack(side=tk.LEFT)
        self.test_label = ttk.Label(top, text="0.20")
        self.test_label.pack(side=tk.LEFT, padx=(4, 0))
        Tooltip(self.test_slider,"Define a porcentagem dos dados que será separada para 'testar' o modelo "
        "\napós o treinamento, ex: 0.20 = 20%. O restante (80%), será usado para 'treinar' "
        "\no modelo.")



        # ---------- Middle bar (barra do meio) ----------
        middle = ttk.Frame(self)
        middle.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Frame dos hiperparâmetros, coluna 0
        hp_frame = ttk.LabelFrame(middle, text="Hiperparâmetros básicos")
        hp_frame.grid(row=0, column=0, padx=10, pady=(10, 10), sticky="nsew")

        # Novo Frame para Pré-processamento (Outliers)
        preprocess_frame = ttk.LabelFrame(middle, text="Tratamento Outliers")
        preprocess_frame.grid(row=0, column=1, padx=10, pady=(10, 10), sticky="nsew")

        ttk.Label(preprocess_frame, text="Outliers:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=5)
        self.outlier_method_cb = ttk.Combobox(preprocess_frame,
        values=["Nenhum", "IQR","Winsorize", "Isolation Forest"],
        state="disabled", width=15)
        self.outlier_method_cb.current(0)
        self.outlier_method_cb.grid(row=0, column=1, sticky=tk.W, padx=4, pady=5)
        Tooltip(self.outlier_method_cb, "Técnica para tratar valores extremos (outliers) que podem prejudicar "
        "\no modelo.\n"
        "\n- IQR: Método estatístico clássico para identificar e corrigir outliers."
        "\n- Winsorize: 'Apara' os valores mais extremos, substituindo-os."
        "\n- Isolation Forest: Algoritmo de Machine Learning para detectar anomalias.")

        ttk.Label(hp_frame, text="n_estimators:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.n_estimators = tk.IntVar(value=100)
        self.n_estimators_entry = ttk.Entry(hp_frame, textvariable=self.n_estimators, width=8)
        self.n_estimators_entry.grid(row=0, column=1, padx=4)

        # Adicionando o tooltip para N Estimators
        Tooltip(self.n_estimators_entry, "Número de árvores na floresta (para RandomForest/XGBoost). "
        "\nValores maiores podem melhorar a precisão, mas aumentam o tempo de treino.")


        ttk.Label(hp_frame, text="max_depth:").grid(row=0, column=2, sticky=tk.W, padx=4)
        self.max_depth = tk.IntVar()
        self.max_depth_entry = ttk.Entry(hp_frame, textvariable=self.max_depth, width=6)
        self.max_depth_entry.grid(row=0, column=3, padx=4)

        # Adicionando o tooltip para Max Depth
        Tooltip(self.max_depth_entry,"Profundidade máxima da árvore. Se for 'None', significa que os nós são expandidos até que todas as folhas sejam puras.")


        ttk.Label(hp_frame, text="learning_rate:").grid(row=0, column=4, sticky=tk.W, padx=4)
        self.lr = tk.DoubleVar()
        self.lr_entry = ttk.Entry(hp_frame, textvariable=self.lr, width=6)
        self.lr_entry.grid(row=0, column=5, padx=4)
        # Adicionando o tooltip para Learning Rate
        Tooltip(self.lr_entry,
                "Taxa de aprendizado (para XGBoost). Controla o peso de cada nova árvore. \nValores menores exigem mais árvores.")

        # Frame de Configuração do Modelo
        model_config_frame = ttk.LabelFrame(middle, text="Configuração do Modelo")
        model_config_frame.grid(row=0, column=2, padx=10, pady=(10, 10), sticky="nsew")

        # Label "Modelos"
        ttk.Label(model_config_frame, text="Modelos:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=5)

        # Combobox "Modelos"
        self.model_cb = ttk.Combobox(model_config_frame,
                                     values=["XGBoost", "RandomForest", "SVM", "KNN", "Rede Neural"], state="readonly",
                                     width=16)
        self.model_cb.current(0)
        self.model_cb.grid(row=0, column=1, sticky=tk.W, padx=4, pady=5)
        self.model_cb.bind("<<ComboboxSelected>>", self.on_model_change)
        Tooltip(self.model_cb, "Escolha o algoritmo de Machine Learning que será treinado.")

        # Checkbutton "Tuning automático"
        self.tune_var = tk.BooleanVar(value=False)
        self.tune_check = ttk.Checkbutton(
            model_config_frame, text="Tuning automático (RandomSearch)", variable=self.tune_var,
            command=self.toggle_cv_combobox
        )
        self.tune_check.grid(row=0, column=2, sticky=tk.W, padx=(12, 4), pady=5)
        Tooltip(self.tune_check,
                "Deixe o software encontrar a melhor combinação de parâmetros para o modelo automaticamente. "
                "\nEle testará várias configurações e escolherá a que tiver o melhor desempenho. Isso pode demorar.")

        # Label "CV" + Combobox de folds — desabilitado até marcar "tuning"
        ttk.Label(model_config_frame, text="CV:").grid(row=0, column=3, sticky=tk.W, padx=4, pady=5)
        self.cv_folds_var = tk.IntVar(value=3)
        self.cv_cb = ttk.Combobox(model_config_frame, textvariable=self.cv_folds_var,
                                  values=[str(i) for i in [3, 5, 10]], width=5, state="disabled")
        self.cv_cb.grid(row=0, column=4, sticky=tk.W, padx=4, pady=5)
        Tooltip(self.cv_cb,
                "Define o número de 'fatias' (folds) para a validação durante o tuning. Cada combinação de parâmetro será avaliada essa quantidade de vezes "
                "\npara garantir que a escolha do 'melhor modelo' seja confiável.")


        # --- Parâmetros do KNN (mesmos widgets/ordem) ---
        ttk.Label(hp_frame, text="n_neighbors:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(10, 0))
        self.knn_n_neighbors_spin = ttk.Spinbox(hp_frame, from_=1, to=30, textvariable=self.knn_n_neighbors, width=6)
        self.knn_n_neighbors_spin.grid(row=1, column=1, padx=4, pady=(10, 0))

        # Adicionando o tooltip para N Neighbors
        Tooltip(self.knn_n_neighbors_spin,"Número de vizinhos a serem usados por padrão para as previsões. Valores maiores podem suavizar as fronteiras de decisão, \nenquanto valores menores podem capturar nuances nos dados.")


        ttk.Label(hp_frame, text="Weights:").grid(row=1, column=2, sticky=tk.W, padx=4, pady=(10, 0))
        self.knn_weights_cb = ttk.Combobox(hp_frame, values=["uniform", "distance"], textvariable=self.knn_weights, state="readonly", width=10)
        self.knn_weights_cb.grid(row=1, column=3, padx=4, pady=(10, 0))
        Tooltip(self.knn_weights_cb,
                "Define o peso dos vizinhos. 'uniform' (todos os pontos têm peso igual) "
                "ou 'distance' (pontos mais próximos têm maior influência).")

        ttk.Label(hp_frame, text="Algorithm:").grid(row=1, column=4, sticky=tk.W, padx=4, pady=(10, 0))
        self.knn_algorithm_cb = ttk.Combobox(hp_frame, values=["auto", "ball_tree", "kd_tree", "brute"],
        textvariable=self.knn_algorithm, state="readonly", width=10)
        self.knn_algorithm_cb.grid(row=1, column=5, padx=4, pady=(10, 0))
        Tooltip(self.knn_algorithm_cb,"Quando 'auto', tentará decidir o melhor algoritmo. Se for 'ball_tree' e 'kd_tree' são mais rápidos para dados grandes, "
        "\nenquanto 'brute' (força bruta) é exato, mas lento.")


        ttk.Label(hp_frame, text="p:").grid(row=2, column=0, sticky=tk.W, padx=4)
        self.knn_p_spin = ttk.Spinbox(hp_frame, from_=1, to=5, textvariable=self.knn_p, width=6)
        self.knn_p_spin.grid(row=2, column=1, padx=4)
        Tooltip(self.knn_p_spin,"Define como a distância entre os pontos é medida. Use 1 para 'distância de Manhattan' (soma das diferenças) "
        "\nou 2 para 'distância Euclidiana' (linha reta).")


        ttk.Label(hp_frame, text="leaf_size:").grid(row=2, column=2, sticky=tk.W, padx=4)
        self.knn_leaf_size_spin = ttk.Spinbox(hp_frame, from_=10, to=100, textvariable=self.knn_leaf_size, width=6)
        self.knn_leaf_size_spin.grid(row=2, column=3, padx=4)
        Tooltip(self.knn_leaf_size_spin,"Usado por 'ball_tree' e 'kd_tree'. Valores menores podem acelerar a predição, mas aumentam o tempo de treino e o uso de memória.")



        # Ações: Treinar e Sair
        action_frame = ttk.Frame(middle)
        action_frame.grid(row=0, column=6, padx=12)

        btn_treinar = tk.Button(
            action_frame, text="Treinar", bg="#000000", fg="#FFFF00",
            activebackground="#363636", relief="flat", font=("Arial", 10, "bold"),
            width=10, command=self.train_button
        )
        btn_treinar.pack(side=tk.LEFT, padx=(0, 5))
        btn_treinar.bind("<Enter>", lambda e: btn_treinar.config(bg="#363636"))
        btn_treinar.bind("<Leave>", lambda e: btn_treinar.config(bg="#000000"))

        # Botão "Salvar Dataset" adicionado aqui
        btn_salvar_dataset = tk.Button(
            action_frame, text="Salvar Dataset", bg="#007BFF", fg="white",
            activebackground="#0056b3", relief="flat", font=("Arial", 10, "bold"),
            width=15, command=self.salvar_dataset
        )
        btn_salvar_dataset.pack(side=tk.LEFT, padx=5)
        btn_salvar_dataset.bind("<Enter>", lambda e: btn_salvar_dataset.config(bg="#0056b3"))
        btn_salvar_dataset.bind("<Leave>", lambda e: btn_salvar_dataset.config(bg="#007BFF"))

        btn_sair = tk.Button(
            action_frame, text="Sair", bg="#FF0000", fg="white",
            activebackground="#FF6347", relief="flat", font=("Arial", 10, "bold"),
            width=10, command=self.force_exit
        )
        btn_sair.pack(side=tk.LEFT)
        btn_sair.bind("<Enter>", lambda e: btn_sair.config(bg="#FF6347"))
        btn_sair.bind("<Leave>", lambda e: btn_sair.config(bg="#FF0000"))

        # Frame da Validação Cruzada (AGORA no lugar onde estava o nn_frame, ao lado do Folds k)
        cv_frame = ttk.Frame(middle)
        cv_frame.grid(row=1, column=0, sticky="w", padx=(15, 0))

        self.use_cv_var = tk.BooleanVar(value=False)
        self.use_cv_cb = ttk.Checkbutton(cv_frame, text="Usar Validação Cruzada", variable=self.use_cv_var)
        self.use_cv_cb.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        Tooltip(self.use_cv_cb,"Para uma avaliação mais confiável do modelo. Em vez de um único teste (que pode ter sorte ou azar), o modelo é treinado e testado várias vezes em diferentes 'fatias' dos dados. "
        "\nO resultado é uma média de desempenho mais estável e realista.")

        self.use_cv_var.trace_add("write", self.on_cv_checkbox_change)

        ttk.Label(cv_frame, text="Métrica para CV:").grid(row=0, column=1, sticky=tk.W)

        self.cv_metric_var = tk.StringVar(value="accuracy")
        self.cv_metric_cb = ttk.Combobox(cv_frame, textvariable=self.cv_metric_var, state="readonly", width=15)
        self.cv_metric_cb["values"] = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        self.cv_metric_cb.grid(row=0, column=2, sticky=tk.W, padx=(5, 10))

        Tooltip(self.cv_metric_cb,"Escolha como o modelo será julgado.\n"
        "\n- Acurácia: % de acertos totais."
        "\n- F1-Score: Equilíbrio para classes desiguais."
        "\n- Precisão: Foco em evitar falsos positivos."
        "\n- Recall: Foco em encontrar todos os positivos.")



        ttk.Label(cv_frame, text="Folds (k):").grid(row=0, column=3, sticky=tk.W)
        self.cv_k_folds = tk.IntVar(value=5)
        self.cv_folds_spin = ttk.Spinbox(cv_frame, from_=2, to=20, textvariable=self.cv_k_folds, width=5)
        self.cv_folds_spin.grid(row=0, column=4, sticky=tk.W, padx=(5, 0))
        Tooltip(self.cv_folds_spin,"Número de 'partes' (k) para dividir os dados. O modelo treina em K-1 partes e valida na restante, "
        "\nrepetindo o processo 'K' vezes para uma avaliação de desempenho mais estável.")


        # Frame da Rede Neural (ABAIXO do hp_frame, mesma linha do cv_frame antigo)
        nn_frame = ttk.LabelFrame(middle, text="Parâmetros - Rede Neural")
        nn_frame.grid(row=1, column=1, columnspan=6, sticky="w", padx=8, pady=(20, 20))

        ttk.Label(nn_frame, text="Hidden Layer:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.nn_hidden_layer = tk.StringVar(value="")  # começa vazio
        self.nn_hidden_entry = ttk.Entry(nn_frame, textvariable=self.nn_hidden_layer, width=8)
        self.nn_hidden_entry.grid(row=0, column=1, padx=4)
        Tooltip(self.nn_hidden_entry,"Define a arquitetura da rede neural. "
        "\nEx: '100' para uma camada com 100 neurônios, ou '100, 50' para duas camadas.")


        ttk.Label(nn_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W, padx=4)
        self.nn_lr = tk.StringVar(value="")  # começa vazio
        self.nn_lr_entry = ttk.Entry(nn_frame, textvariable=self.nn_lr, width=8)
        self.nn_lr_entry.grid(row=0, column=3, padx=4)
        Tooltip(self.nn_lr_entry,"Controla a velocidade de aprendizado. Valores pequenos (ex: 0.001) são mais lentos e precisos. "
        "\nValores grandes (ex: 0.1) podem acelerar, mas podem apresentar instabilidade.")


        ttk.Label(nn_frame, text="Max Iter:").grid(row=0, column=4, sticky=tk.W, padx=4)
        self.nn_max_iter = tk.StringVar(value="")  # começa vazio
        self.nn_max_iter_entry = ttk.Entry(nn_frame, textvariable=self.nn_max_iter, width=8)
        self.nn_max_iter_entry.grid(row=0, column=5, padx=4)
        Tooltip(self.nn_max_iter_entry,"Número máximo de 'passos' que o treino executará. Se for necessário, "
        "\naumente se o modelo não estiver aprendendo o suficiente. Valor padrão: 300")


        # --- NOVO parâmetro Rede Neural na linha 1
        ttk.Label(nn_frame, text="Momentum:").grid(row=1, column=0, sticky=tk.W, padx=4)
        self.nn_momentum = tk.StringVar(value="")  # começa vazio
        self.nn_momentum_entry = ttk.Entry(nn_frame, textvariable=self.nn_momentum, width=8)
        self.nn_momentum_entry.grid(row=1, column=1, padx=4)
        Tooltip(self.nn_momentum_entry,"Ajuda o otimizador 'sgd' a acelerar o aprendizado e evitar mínimos locais. Usado apenas com o solver 'sgd'. "
        "\nValor comum: entre 0.0 a 0.9.")


        ttk.Label(nn_frame, text="Solver:").grid(row=1, column=2, sticky=tk.W, padx=4)
        self.nn_solver = tk.StringVar(value="adam")  # valor padrão
        self.nn_solver_combo = ttk.Combobox(nn_frame, textvariable=self.nn_solver, width=8, state="readonly")
        self.nn_solver_combo['values'] = ("adam", "sgd", "lbfgs")
        self.nn_solver_combo.grid(row=1, column=3, padx=(0, 4))
        Tooltip(self.nn_solver_combo,"Algoritmo para otimizar os pesos. 'adam' é robusto para a maioria dos casos. "
        "\nO 'sgd' é mais simples, enquanto o 'lbfgs' é eficiente para datasets pequenos.")

        self.nn_solver_combo.bind("<<ComboboxSelected>>", self.on_solver_change)

        # Desabilita os campos da Rede Neural por padrão
        for widget in [self.nn_hidden_entry,
                       self.nn_max_iter_entry,
                       self.nn_lr_entry,
                       self.nn_solver_combo,
                       self.nn_momentum_entry]:
            widget.config(state="disabled")

        # ---------- Esquerda: Tabela ----------
        left = ttk.Frame(self, width=900)  # largura fixa inicial (ajuste conforme a tela)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)
        left.pack_propagate(False)  # impede expansão automática

        ttk.Label(left, text="Pré-visualização dos dados").pack(anchor=tk.W)
        self.tree_frame = ttk.LabelFrame(left, text="Visualização do Arquivo Carregado")
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        # Impede que o LabelFrame seja redimensionado automaticamente pelos filhos
        self.tree_frame.pack_propagate(False)
        # Define altura inicial razoável para a tabela
        self.tree_frame.configure(height=420)

        # Treeview
        self.tree = ttk.Treeview(self.tree_frame, show="headings")

        # Scrollbars
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)

        # Grid
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configuração das barras na Tree
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Faz o frame respeitar os pesos (expansão controlada)
        self.tree_frame.rowconfigure(0, weight=1)
        self.tree_frame.columnconfigure(0, weight=1)

        # Binding para duplo clique na Treeview
        self.tree.bind("<Double-1>", self.on_double_click)

        # ---------- Direita: Métricas + Gráficos ----------
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        metrics_frame = ttk.LabelFrame(right, text="Resultados")
        metrics_frame.pack(fill=tk.X)

        self.acc_label = ttk.Label(metrics_frame, text="Acurácia: -", font=("Arial", 10, "bold"))
        self.acc_label.pack(anchor=tk.W, padx=6, pady=2)

        self.acc_spinner_label = ttk.Label(metrics_frame, text="", font=("Arial", 10, "bold"))
        self.acc_spinner_label.pack(anchor=tk.W, padx=6)

        self.f1_label = ttk.Label(metrics_frame, text="F1-score: -")
        self.f1_label.pack(anchor=tk.W, padx=6, pady=2)

        self.cm_label = ttk.Label(metrics_frame, text="Matriz de confusão: -")
        self.cm_label.pack(anchor=tk.W, padx=6, pady=2)

        plot_frame = ttk.LabelFrame(right, text="Gráficos")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(6, 3))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_pred = tk.Button(
            right, text="Fazer Predição Manual", bg="#008B8B", fg="white",
            activebackground="#20B2AA", font=("Arial", 11, "bold"), relief="flat",
            width=22, command=self.predicao_manual
        )
        btn_pred.pack(pady=6)
        btn_pred.bind("<Enter>", lambda e: btn_pred.config(bg="#20B2AA"))
        btn_pred.bind("<Leave>", lambda e: btn_pred.config(bg="#008B8B"))

        # Status do treinamento
        self.train_status = ttk.Label(right, text="Máquina Não Treinada", foreground="#F21707",
                                      font=("Arial", 11, "bold"))
        self.train_status.pack(pady=4)

        # Barra de status inferior

        # Bind global p/ clique
        self.bind("<Button-1>", self.click_geral)

        # Ajustes iniciais conforme modelo padrão
        self.on_model_change()
        self.on_cv_checkbox_change()  # -> DESABILITA PARAMETROS DA VALIDACAO CRUZADA NA INICLIAZACAO do app

    # =============================================================================
    # 9) AJUSTES/UTILITÁRIOS DE CONTROLE (UTILS)
    # =============================================================================

    def edit_row(self, row_index: int) -> None:
        """Abre uma janela para editar o registro na linha especificada."""
        top = tk.Toplevel(self)
        top.title(f"Editar Registro {row_index}")
        top.geometry("600x700")
        top.resizable(False, True)
        top.iconbitmap("machine_learning.ico")

        # Obtém os valores da linha do DataFrame
        row_data = self.df.iloc[row_index].copy()
        entries = {}

        # Frame principal para conter o Canvas e a barra de rolagem
        main_frame = ttk.Frame(top)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Canvas para os campos roláveis
        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Barra de rolagem vertical
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        vsb.pack(side=tk.RIGHT, fill="y")
        canvas.configure(yscrollcommand=vsb.set)

        # Frame interno no Canvas para os widgets
        scrollable_frame = ttk.Frame(canvas)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Título (centralizado)
        ttk.Label(
            scrollable_frame,
            text=f"Editar Registro {row_index}",
            font=("Arial", 12, "bold"),
            anchor="center"
        ).grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky="ew")

        # Cria campos de entrada para cada coluna usando grid
        for i, col in enumerate(self.df.columns, start=1):
            # Label à esquerda
            label = ttk.Label(scrollable_frame, text=col, font=("Arial", 10))
            label.grid(row=i, column=0, padx=(0, 10), pady=5, sticky="e")  # Alinhado à direita

            # Entry à direita
            ent = ttk.Entry(scrollable_frame, width=30)
            ent.insert(0, str(row_data[col]))  # Preenche com o valor atual
            ent.grid(row=i, column=1, padx=(0, 10), pady=5, sticky="w")  # Alinhado à esquerda
            entries[col] = ent

        # Configura pesos para centralizar o conteúdo
        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)

        def save_changes():
            """Salva as alterações no DataFrame e atualiza a Treeview."""
            try:
                # Coleta os novos valores
                new_values = {}
                for col, ent in entries.items():
                    val = ent.get().strip()
                    # Converte para o tipo apropriado com base na coluna
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        try:
                            val = float(val) if pd.api.types.is_float_dtype(self.df[col]) else int(val)
                        except ValueError:
                            raise ValueError(f"Valor inválido para a coluna '{col}': deve ser numérico.")
                    new_values[col] = val

                # Atualiza o DataFrame
                for col, val in new_values.items():
                    self.df.at[row_index, col] = val

                # Atualiza a Treeview
                self.populate_table()

                # Atualiza o rótulo do arquivo carregado
                file_name = os.path.basename(self.file_path) if self.file_path else "Dataset"
                self.loaded_file_label.config(text=f"{file_name} ({len(self.df)} linhas)", foreground="green")

                # Atualiza a contagem de classes, se necessário
                self.on_target_change()

                messagebox.showinfo("Sucesso", "Registro atualizado com sucesso!")
                top.destroy()

            except Exception as e:
                messagebox.showerror("Erro - Edição", f"Erro ao salvar alterações: {self.traduzir_erro(str(e))}")

        # Frame para os botões (fora do Canvas, na parte inferior)
        button_frame = ttk.Frame(top)
        button_frame.pack(fill=tk.X, padx=15, pady=(10, 15))

        # Botão para salvar alterações
        btn_save = tk.Button(
            button_frame,
            text="Salvar Alterações",
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            relief="flat",
            font=("Arial", 11, "bold"),
            width=15,
            command=save_changes
        )
        btn_save.pack(pady=5)
        btn_save.bind("<Enter>", lambda e: btn_save.config(bg="#5DD75D"))
        btn_save.bind("<Leave>", lambda e: btn_save.config(bg="#4CAF50"))

        # Botão para cancelar
        btn_cancel = tk.Button(
            button_frame,
            text="Cancelar",
            bg="#FF0000",
            fg="white",
            activebackground="#FF6347",
            relief="flat",
            font=("Arial", 11, "bold"),
            width=15,
            command=top.destroy
        )
        btn_cancel.pack(pady=5)
        btn_cancel.bind("<Enter>", lambda e: btn_cancel.config(bg="#FF6347"))
        btn_cancel.bind("<Leave>", lambda e: btn_cancel.config(bg="#FF0000"))

        # Atualiza a região rolável do Canvas
        def configure_canvas(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_frame, width=canvas.winfo_width())

        scrollable_frame.bind("<Configure>", configure_canvas)

        # Suporte para rolagem com a roda do mouse
        def on_mouse_wheel(event):
            if top.winfo_exists():  # Verifica se a janela ainda existe
                canvas.yview_scroll(-1 * (event.delta // 120), "units")

        # Vincula o evento <MouseWheel> apenas à janela Toplevel
        top.bind("<MouseWheel>", on_mouse_wheel)

        # Remove o binding quando a janela for destruída
        def on_window_destroy(event):
            top.unbind("<MouseWheel>")

        top.bind("<Destroy>", on_window_destroy)

    def on_double_click(self, event) -> None:
        """Captura o duplo clique na Treeview e abre janela de edição."""
        if self.df is None:
            messagebox.showerror("Erro - DataSet", "Carregue um dataset primeiro!")
            return

        # Identifica a região clicada
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return  # Ignora cliques fora de células

        # Obtém o item (linha) selecionado
        item = self.tree.identify_row(event.y)
        if not item:
            return  # Nenhum item selecionado

        # Obtém o índice da linha no DataFrame
        # Como a Treeview mostra apenas as primeiras 80 linhas, usamos o index do item
        item_index = int(self.tree.index(item))
        if item_index >= len(self.df):
            return  # Índice inválido

        # Abre janela de edição para o registro
        self.edit_row(item_index)

    def on_target_change(self, event=None):
        """Atualiza a visualização quando o alvo é trocado."""
        if self.df is None or not self.target_cb.get():
            self.target_count_label.config(text="Contagem de classes: -")
            return

        target = self.target_cb.get()
        contagem = self.df[target].value_counts()
        texto = "Contagem de classes: " + ", ".join([f"{k}: {v}" for k, v in contagem.items()])
        self.target_count_label.config(text=texto)

        if self.view_mode == "distribuicao_alvo":
            self.exibir_distribuicao_alvo()
        elif all(getattr(self, attr, None) is not None for attr in ["acc", "f1", "cm", "y_test", "y_pred", "model"]):
            self.update_results(self.acc, self.f1, self.cm, self.y_test, self.y_pred, self.model)

    def exibir_visualizacao_original(self) -> None:
        """Restaura a visualização padrão no frame direito (matriz de confusão + ROC ou CV)."""
        # Verifica se já está na visualização original
        if self.view_mode == "original":
            messagebox.showinfo("Aviso - Métricas", "Você já está visualizando as métricas originais.")
            return  # Não faz nada, pois já está na visualização original

        # Caso contrário, continua com o fluxo de exibição normal
        self.view_mode = "original"

        # Verifica se existem resultados disponíveis para exibir
        if all(getattr(self, attr, None) is not None for attr in ["acc", "f1", "cm", "y_test", "y_pred", "model"]):
            self.update_results(self.acc, self.f1, self.cm, self.y_test, self.y_pred, self.model)
        else:
            # Se não houver resultados, limpa os gráficos
            for ax in self.axs:
                ax.clear()
            self.fig.tight_layout()
            self.canvas.draw()
            self.canvas.get_tk_widget().update_idletasks()

    def exibir_distribuicao_alvo(self) -> None:
        """Exibe a distribuição da variável alvo em gráfico de barras horizontal (no frame direito)."""
        if self.df is None:
            messagebox.showwarning("Aviso - Carregar DataSet",
                                   "Carregue um DataSet antes de visualizar a distribuição.")
            return

        target = self.target_cb.get()
        if not target:
            messagebox.showwarning("Aviso - Selecione Coluna",
                                   "Selecione a coluna alvo para visualizar a distribuição.")
            return

        self.view_mode = "distribuicao_alvo"

        try:
            # Conta ocorrências das classes
            contagem = self.df[target].value_counts().sort_index()

            # Limpa os gráficos anteriores
            for ax in self.axs:
                ax.clear()

            ax = self.axs[0]  # Usa apenas o primeiro gráfico do painel

            total = contagem.sum()
            classe_mais_comum = contagem.idxmax()
            classe_menos_comum = contagem.idxmin()

            # Define as cores conforme a classe
            colors = []
            for cls in contagem.index:
                if cls == classe_mais_comum:
                    colors.append("#2BEB09")  # verde
                elif cls == classe_menos_comum:
                    colors.append("#D9150D")  # vermelho
                else:
                    colors.append("#4682B4")  # azul padrão

            # Cria o gráfico horizontal
            bars = ax.barh(contagem.index.astype(str), contagem.values, color=colors, height=0.3)


            ax.set_title("Distribuição da coluna selecionada", fontsize=11)
            ax.set_xlabel("Ocorrências", fontsize=10)
            ax.set_ylabel("Classe", fontsize=10)

            # Adiciona o valor absoluto e a porcentagem nas barras
            for bar, valor in zip(bars, contagem.values):
                largura = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2

                # Valor absoluto à direita da barra
                ax.text(largura + max(contagem.values) * 0.01, y, str(valor),
                        ha='left', va='center', fontsize=9)

                # Porcentagem dentro da barra
                perc = valor / total * 100
                ax.text(
                    largura / 2,
                    y,
                    f"{perc:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white"
                )

            # Painel direito: texto explicativo
            self.axs[1].clear()
            ocorrencias_mais_comum = contagem[classe_mais_comum]
            ocorrencias_menos_comum = contagem[classe_menos_comum]
            porcentagem_mais_comum = ocorrencias_mais_comum / total * 100
            porcentagem_menos_comum = ocorrencias_menos_comum / total * 100

            texto = (
                f"Classe mais comum: '{classe_mais_comum}'\n"
                f"Ocorrências: {ocorrencias_mais_comum} ({porcentagem_mais_comum:.2f}%)\n\n"
                f"Classe menos comum: '{classe_menos_comum}'\n"
                f"Ocorrências: {ocorrencias_menos_comum} ({porcentagem_menos_comum:.2f}%)\n\n"
                f"Total: {total}"
            )

            self.axs[1].text(
                0.5, 0.5,
                texto,
                ha="center", va="center", fontsize=11, wrap=True,
                bbox=dict(facecolor="#f0f0f0", edgecolor="black")
            )
            self.axs[1].axis("off")

            self.fig.tight_layout()
            self.canvas.draw()
            self.canvas.get_tk_widget().update_idletasks()

        except Exception as e:
            messagebox.showerror("Erro - Gráfico", f"Não foi possível exibir a distribuição.\nErro: {e}")

    def salvar_dataset(self) -> None:
        """
        Salva o dataset atual em um arquivo, em qualquer extensão suportada.
        """
        if self.df is None:
            messagebox.showerror("Erro - Carregue DataSet", "Não há um dataset carregado para salvar.")
            return

        file_types = [
            ("Arquivos CSV", "*.csv"),
            ("Arquivos Excel", "*.xlsx"),
            ("Arquivos de Texto", "*.txt"),
            ("Todos os arquivos", "*.*")
        ]
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=file_types,
            title="Salvar Dataset como"
        )
        if not save_path:
            return

        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".csv":
                self.df.to_csv(save_path, index=False)
            elif ext == ".xlsx":
                self.df.to_excel(save_path, index=False)
            elif ext == ".txt":
                self.df.to_csv(save_path, index=False, sep="\t")  # Exemplo de separador por tab
            else:
                messagebox.showerror("Erro - Arquivo", f"Extensão de arquivo não suportada: {ext}")
                return

            messagebox.showinfo("Salvo com sucesso!", f"Dataset salvo em: {save_path}")

        except Exception as e:
            messagebox.showerror("Erro - Salvar", f"Ocorreu um erro ao salvar o arquivo: {e}")

    def on_solver_change(self, event=None):
        solver = self.nn_solver.get()
        if solver == 'sgd':
            self.nn_momentum_entry.config(state='normal')
        else:
            self.nn_momentum_entry.config(state='disabled')
            self.nn_momentum.set('')  # limpa o campo quando desabilita

    def _unfocus_entries(self, widget) -> None:
        """Remove o foco de campos Entry em toda a árvore de widgets."""
        if isinstance(widget, ttk.Entry) and widget == self.focus_get():
            self.focus_set()
        for child in widget.winfo_children():
            self._unfocus_entries(child)

    def _unfocus_comboboxes(self, widget) -> None:
        """Remove o foco de Comboboxes em toda a árvore de widgets."""
        if isinstance(widget, ttk.Combobox) and widget == self.focus_get():
            self.focus_set()
        for child in widget.winfo_children():
            self._unfocus_comboboxes(child)

    def _update_test_label(self, val: str) -> None:
        """Atualiza label com o valor atual do slider de test_size."""
        v = float(val)
        self.test_size_var.set(round(v, 2))
        self.test_label.config(text=f"{v:.2f}")

    def on_model_change(self, event=None) -> None:
        """Habilita/desabilita controles conforme o modelo selecionado (layout preservado)."""
        model = self.model_cb.get()

        def set_state_knn(state: str) -> None:
            for widget in [
                self.knn_n_neighbors_spin, self.knn_weights_cb, self.knn_algorithm_cb,
                self.knn_p_spin, self.knn_leaf_size_spin
            ]:
                widget.config(state=state)

        def set_state_generic_hp(state: str) -> None:
            for widget in [self.n_estimators_entry, self.max_depth_entry, self.lr_entry]:
                widget.config(state=state)

        def set_state_nn(state: str) -> None:
            for widget in [
                self.nn_hidden_entry,
                self.nn_max_iter_entry,
                self.nn_lr_entry,
                self.nn_momentum_entry,
                self.nn_solver_combo
            ]:
                widget.config(state=state)

        # Desativa tudo inicialmente
        set_state_knn("disabled")
        set_state_generic_hp("disabled")
        set_state_nn("disabled")

        # Ativa somente o que for necessário
        if model == "KNN":
            set_state_knn("normal")
            # Restaura valores padrão do KNN
            self.knn_n_neighbors.set(5)
            self.knn_weights.set("uniform")
            self.knn_algorithm.set("auto")
            self.knn_p.set(2)
            self.knn_leaf_size.set(30)


        elif model == "Rede Neural":

            set_state_nn("normal")

            # Limpa os campos da Rede Neural

            self.nn_hidden_entry.delete(0, 'end')
            self.nn_lr_entry.delete(0, 'end')
            self.nn_max_iter_entry.delete(0, 'end')
            self.nn_momentum_entry.delete(0, 'end')
            self.nn_solver_combo.current(0)

            # Atualiza estado do momentum de acordo com o solver atual
            self.on_solver_change()


        elif model in ["XGBoost", "RandomForest"]:
            set_state_generic_hp("normal")

    def force_exit(self) -> None:
        """Fecha imediatamente o processo (mantido conforme original)."""
        os._exit(0)

    def toggle_cv_combobox(self) -> None:
        """Habilita/desabilita seleção de folds de CV com base no check de tuning."""
        self.cv_cb.configure(state="readonly" if self.tune_var.get() else "disabled")
        if self.tune_var.get():
            self.n_estimators_entry.config(state="disabled")
            self.max_depth_entry.config(state="disabled")
            self.lr_entry.config(state="disabled")
        else:
            self.n_estimators_entry.config(state="normal")
            self.max_depth_entry.config(state="normal")
            self.lr_entry.config(state="normal")

    # =============================================================================
    # 10) ABERTURA DE ARQUIVO
    # =============================================================================
    def open_file(self) -> None:
        """Abre arquivo de dados (csv/xlsx/txt/data/sqlite) e popula a Treeview."""
        ftypes = [
            ("All files", "*.*"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("Text files", "*.txt;*.tsv"),
            ("Data files", "*.data"),
            ("SQLite DB", "*.db;*.sqlite"),
        ]
        path = filedialog.askopenfilename(filetypes=ftypes)

        # Limpa informações anteriores para aceitar novo dataset
        self.df = None
        self.file_path = None
        self.target_cb.set("")
        self.target_cb["values"] = []

        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        detected_type = None

        try:
            if ext == ".csv":
                df = pd.read_csv(path)
                detected_type = "CSV"
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
                detected_type = "XLSX"
            elif ext in [".txt", ".tsv", ".data"]:
                df = pd.read_csv(path, sep=None, engine="python")
                detected_type = "TXT" if ext != ".csv" else "CSV"
            elif ext in [".db", ".sqlite"]:
                conn = sqlite3.connect(path)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [r[0] for r in cur.fetchall()]
                conn.close()
                if not tables:
                    messagebox.showerror("Erro - Banco de Dados", "Banco SQLite sem tabelas.")
                    return
                table = tables[0] if len(tables) == 1 else self.ask_table(tables)
                if not table:
                    return
                conn = sqlite3.connect(path)
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                conn.close()
                detected_type = "SQLite"
            else:
                messagebox.showwarning("Aviso - Formato", f"Formato {ext} não suportado.")
                return

        except Exception as e:
            messagebox.showerror("Erro - Leitura Arquivo", f"Falha ao ler arquivo: {e}")
            return

        # Atualiza tipo automaticamente no combobox
        if detected_type is not None:
            self.file_type.set(detected_type)

        # Salva e mostra os dados
        self.df = df
        self.populate_table()

        # Preenche combobox da coluna alvo
        if not df.empty:
            self.target_cb["values"] = list(df.columns)
            self.target_cb.set("")
            self.outlier_method_cb.config(state="readonly") #habilita combobox outliers

        file_name = os.path.basename(path)
        self.loaded_file_label.config(text=f"{file_name} ({len(df)} linhas)", foreground="green")
        self.file_path = path

    def validate_file_type_change(self, event) -> None:
        """Garante que o tipo selecionado combine com a extensão do arquivo carregado."""
        if self.df is None or not self.file_path:
            return

        selected_type = self.file_type.get()
        ext = os.path.splitext(self.file_path)[1].lower()

        if ext == ".csv":
            detected_type = "CSV"
        elif ext in [".xlsx", ".xls"]:
            detected_type = "XLSX"
        elif ext in [".txt", ".tsv", ".data"]:
            detected_type = "TXT"
        elif ext in [".db", ".sqlite"]:
            detected_type = "SQLite"
        else:
            detected_type = "Desconhecido"

        if selected_type != detected_type:
            messagebox.showerror(
                "Erro - Arquivo Incorreto",
                f"Você selecionou '{selected_type}', mas o arquivo carregado é do tipo '{detected_type}'.\n"
                f"Irei ajustar a seleção automaticamente."
            )
            self.file_type.set(detected_type)

    def ask_table(self, tables: list[str]) -> str | None:
        """Pergunta ao usuário qual tabela abrir (quando há múltiplas no SQLite)."""
        table = simpledialog.askstring("Escolha a tabela", f"Banco possui várias tabelas:\n{tables}\nDigite a tabela desejada:")
        while table not in tables:
            if table is None:
                return None
            messagebox.showerror("Erro - Tabela", "Tabela inválida. Tente novamente.")
            table = simpledialog.askstring("Escolha a tabela", f"Banco possui várias tabelas:\n{tables}\nDigite a tabela desejada:")
        return table

    # =============================================================================
    # 11) TABELA (TREEVIEW) — POPULAÇÃO E CONTEXTO
    # =============================================================================
    def populate_table(self) -> None:
        """Preenche a Treeview com as primeiras 80 linhas do DataFrame atual."""
        if self.df is None:
            return

        # Limpa dados existentes
        for c in self.tree.get_children():
            self.tree.delete(c)

        self.tree["columns"] = list(self.df.columns)

        # Define cabeçalhos/colunas
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor=tk.W, stretch=False)

        # Insere até 80 linhas
        for r in self.df.head(80).values.tolist():
            self.tree.insert("", tk.END, values=[str(x) for x in r])

        # Menu de contexto no cabeçalho
        if len(self.df.columns) >= 1:
            self.tree.bind("<Button-3>", self.show_column_context_menu)
        else:
            self.tree.unbind("<Button-3>")

    def show_column_context_menu(self, event) -> None:
        """Menu de contexto no cabeçalho para excluir a coluna selecionada."""
        if self.df is None:
            # Exibe uma mensagem de erro ou aviso
            messagebox.showerror("Erro", "Nenhum dataset carregado.")
            return
        region = self.tree.identify_region(event.x, event.y)
        if region == "heading":
            col_id = self.tree.identify_column(event.x)
            col_index = int(col_id.replace("#", "")) - 1
            col_name = self.df.columns[col_index]
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label=f"Excluir coluna: {col_name}", command=lambda: self.remove_column(col_name))
            menu.post(event.x_root, event.y_root)

    def remove_column(self, column_name: str) -> None:
        """Remove coluna do DataFrame e atualiza Treeview/combobox alvo."""
        self.df.drop(columns=[column_name], inplace=True)
        self.populate_table()

        self.target_cb["values"] = list(self.df.columns)
        if self.target_cb.get() == column_name:
            self.target_cb.set("")

        self.tree.bind("<Button-3>", self.show_column_context_menu)

    # =============================================================================
    # 12) TREINO DO MODELO
    # =============================================================================
    def train_button(self) -> None:
        """Valida pré-condições e inicia treinamento em thread separada."""
        self.start_text_spinner()

        if self.df is None:
            messagebox.showwarning("Aviso - Dados", "Carregue um arquivo primeiro.")
            self.stop_text_spinner()
            return
        if not self.target_cb.get():
            messagebox.showwarning("Aviso - Target", "Selecione a coluna alvo.")
            self.stop_text_spinner()
            return

        threading.Thread(target=self.train_model, daemon=True).start()

    def traduzir_erro(self, msg: str) -> str:
        """Traduções amigáveis para mensagens comuns de erro."""
        msg = (msg or "").lower()

        # Numéricos
        if "invalid literal for int" in msg:
            return "O valor não é um número inteiro válido."
        if "could not convert string to float" in msg:
            return "O valor não é um número decimal válido."
        if "must be real number" in msg:
            return "O campo deve conter um número válido."
        if "is not in list" in msg:
            return "O valor selecionado não está na lista de opções."
        if "cannot be negative" in msg or "value must be non-negative" in msg or "negative" in msg:
            return "O valor não pode ser negativo."
        if "must be positive" in msg:
            return "O valor deve ser maior que zero."
        if "int() argument must be a string" in msg:
            return "Preencha o campo com um número inteiro válido."
        if "division by zero" in msg:
            return "Erro: divisão por zero detectada."

        # Pandas / DataFrame
        if "no columns to parse from file" in msg:
            return "O arquivo não contém colunas reconhecíveis."
        if "could not convert string to timestamp" in msg:
            return "O valor não pôde ser convertido para data/hora."
        if "keyerror" in msg:
            return "Coluna não encontrada no DataSet."

        # Treino / ML
        if "found input variables with inconsistent numbers of samples" in msg:
            return "As variáveis de entrada e saída têm tamanhos diferentes."
        if "unknown label type" in msg:
            return "O tipo de variável alvo não é reconhecido."
        if "n_classes" in msg or "number of classes" in msg:
            return "O número de classes no alvo é inválido."
        if (" has " in msg and "features" in msg and (" expect" in msg or " expecting " in msg)):
            return "O número de colunas/features do DataSet não corresponde ao esperado pelo modelo treinado."
        if "feature shape mismatch" in msg and "expected" in msg and "got" in msg:
            return "O número de colunas/features do DataSet não corresponde ao esperado pelo modelo treinado."
        if ("expected" in msg and "got" in msg) and ("feature" in msg or "features" in msg or "attribute" in msg or "attributes" in msg):
            return "O número de colunas/features do DataSet não corresponde ao esperado pelo modelo treinado."

        # NumPy
        if "shapes" in msg and "not aligned" in msg:
            return "As dimensões dos arrays não são compatíveis para a operação."
        if "operands could not be broadcast together" in msg:
            return "As dimensões dos arrays não são compatíveis para a operação."
        if "index out of bounds" in msg:
            return "Índice fora dos limites do array."

        # Arquivos / IO
        if "filenotfounderror" in msg or "no such file or directory" in msg:
            return "Arquivo não encontrado."
        if "permission denied" in msg:
            return "Permissão negada para acessar o arquivo."
        if "unsupported file type" in msg:
            return "Tipo de arquivo não suportado."

        # SQLite
        if "no such table" in msg:
            return "Tabela não encontrada no banco de dados."
        if "database is locked" in msg:
            return "Banco de dados está em uso por outro processo."
        if "syntax error" in msg:
            return "Erro de sintaxe na consulta SQL."

        # Genéricos Python
        if "valueerror" in msg:
            return "Valor inválido fornecido."
        if "typeerror" in msg:
            return "Tipo de dado incorreto."
        if "attributeerror" in msg:
            return "Operação ou atributo inválido para este objeto."

        # Fallback
        return "Verifique esse erro:  " + msg

    def train_model(self) -> None:
        """Pipeline de treino: preparo, split, (opcional) GridSearch, treino, CV e métricas."""
        try:

            df = self.df.copy()
            target = self.target_cb.get()

            # ---------------- X (features) / y ----------------
            X = df.drop(columns=[target])
            y = df[target]

            # --- TRATAMENTO DE OUTLIERS ---
            outlier_method = self.outlier_method_cb.get()
            if outlier_method != "Nenhum":
                numeric_cols = X.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    if outlier_method == "Winsorize":
                        X[numeric_cols] = winsorize_df(X[numeric_cols], verbose=True)
                    elif outlier_method == "IQR":
                        X[numeric_cols] = iqr_clip(X[numeric_cols], verbose=True)
                    elif outlier_method == "Isolation Forest":
                        X = isolation_forest_replace_median(X, verbose=True)
                else:
                    messagebox.showwarning("Tratamento de Outliers",
                                           "Nenhuma coluna numérica foi encontrada para aplicar o tratamento.")

            # --- TRATAMENTO DE VARIÁVEIS CATEGÓRICAS ---
            colunas_descartadas = []
            # Seleciona colunas de texto ou categoria
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns

            for c in categorical_cols:
                # Verifica a quantidade de valores únicos (cardinalidade)
                if X[c].nunique() <= 20:
                    # Se for baixa, converte para códigos numéricos
                    X[c] = X[c].astype("category").cat.codes
                else:
                    # Se for alta, descarta a coluna para evitar problemas de performance/dimensionalidade
                    colunas_descartadas.append(c)

            if colunas_descartadas:
                X = X.drop(columns=colunas_descartadas)
                colunas_str = ", ".join(colunas_descartadas)
                messagebox.showinfo(
                    "Aviso - Colunas Descartadas",
                    f"As seguintes colunas foram removidas devido à alta cardinalidade (mais de 20 categorias):\n\n{colunas_str}"
                )

                # Alvo categórico -> converte para códigos numéricos
                if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                    y = y.astype("category").cat.codes

                # Garante que o DataFrame X contenha apenas colunas numéricas antes de escalar
                X = X.select_dtypes(include=np.number)

            # ---------------- Escalonamento ----------------
            if self.scaler_cb.get() == "MinMax":
                self.scaler = MinMaxScaler((0, 1))
                Xs = self.scaler.fit_transform(X)
            elif self.scaler_cb.get() == "Standard":
                self.scaler = StandardScaler()
                Xs = self.scaler.fit_transform(X)
            else:
                self.scaler = None
                Xs = X.values

            # ---------------- Split treino/teste (estratificado) ----------------
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=self.test_size_var.get(), random_state=7, stratify=y
            )
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

            # ---------------- Seleção do tipo de modelo ----------------
            model_type = self.model_cb.get()

            # Peso automático (XGBoost binário) p/ desbalanceamento
            scale_pos_weight_val = 1
            if model_type == "XGBoost" and len(np.unique(y_train)) == 2:
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                if pos_count > 0:
                    scale_pos_weight_val = neg_count / pos_count

            # Validação de max_depth (apenas em modo manual)
            if not self.tune_var.get() and model_type in ["XGBoost", "RandomForest"]:
                try:
                    max_depth_input = self.max_depth.get()
                    if isinstance(max_depth_input, str) and max_depth_input.strip() == "":
                        raise ValueError("Campo 'max_depth' está vazio.")
                    max_depth_value = int(max_depth_input)
                    if max_depth_value <= 0:
                        raise ValueError("'Max_Depth' deve ser maior que zero.")
                except Exception as e:
                    messagebox.showerror(
                        "Erro - Max_Depth",
                        f"O valor inserido para profundidade máxima é inválido.\n{self.traduzir_erro(str(e))}"
                    )
                    self.after(0, self.stop_text_spinner)
                    return
            else:
                max_depth_value = None

            # ---------------- MODO AUTOMATICO (RandomSearch) ----------------

            if self.tune_var.get():
                # RandomizedSearch por modelo
                if model_type == "XGBoost":
                    model = XGBClassifier(
                        eval_metric="mlogloss",
                        scale_pos_weight=scale_pos_weight_val,
                    )
                    param_grid = {
                        "n_estimators": randint(100, 500),
                        "max_depth": randint(3, 7),
                        "learning_rate": uniform(0.01, 0.2),
                        "subsample": uniform(0.6, 0.4),
                        "colsample_bytree": uniform(0.6, 0.4),
                        "min_child_weight": randint(1, 10),
                        "gamma": uniform(0, 0.5),
                        "reg_alpha": uniform(0, 0.2),
                        "reg_lambda": uniform(1, 2),
                    }

                elif model_type == "RandomForest":
                    model = RandomForestClassifier(
                        class_weight="balanced",
                    )
                    param_grid = {
                        "n_estimators": randint(100, 600),
                        "max_depth": [None, 6, 7, 8],
                        "min_samples_split": randint(2, 20),
                        "min_samples_leaf": randint(1, 10),
                        "bootstrap": [True, False],
                        "criterion": ["gini", "entropy"],
                        "max_features": ["sqrt", "log2", 0.5, 0.75],
                    }
                elif model_type == "SVM":
                    model = SVC(
                        probability=True,
                        class_weight="balanced",
                    )
                    param_grid = {
                        "C": uniform(0.1, 10),
                        "kernel": ["linear", "rbf"],
                        "gamma": ["scale", "auto", uniform(0.001, 0.1)],
                        "degree": [2, 3, 4],
                    }

                elif model_type == "Rede Neural":
                    model = MLPClassifier(
                        random_state=50,
                        early_stopping=True,
                    )
                    param_grid = {
                        "hidden_layer_sizes": [(50,), (100,), (100, 50), (50, 50, 50)],
                        "activation": ["tanh", "relu"],
                        "solver": ["adam", "sgd"],
                        "alpha": uniform(0.0001, 0.1),
                        "learning_rate_init": uniform(0.001, 0.1),
                        "max_iter": randint(200, 500),
                    }

                elif model_type == "KNN":
                    model = KNeighborsClassifier(
                    )
                    param_grid = {
                        "n_neighbors": randint(3, 20),
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree"],
                        "p": [1, 2],
                        "leaf_size": randint(20, 50),
                    }
                else:
                    messagebox.showerror("Erro", f"Modelo '{model_type}' não reconhecido para RandomizedSearch.")
                    self.after(0, self.stop_text_spinner)
                    return

                cv_folds = self.cv_folds_var.get()
                rs = RandomizedSearchCV(model, param_grid, n_iter=30, scoring="accuracy", cv=cv_folds, n_jobs=2,random_state=42)
                rs.fit(X_train, y_train)
                model = rs.best_estimator_
                model.fit(X_train, y_train)
                self.model = model

            else:
                # ---------------- MODO MANUAL ----------------

                if model_type == "XGBoost":
                    model = XGBClassifier(
                        eval_metric="mlogloss",
                        n_estimators=self.n_estimators.get(),
                        max_depth=max_depth_value,
                        scale_pos_weight=scale_pos_weight_val,
                        learning_rate=self.lr.get(),
                    )

                    model.fit(X_train, y_train)

                elif model_type == "RandomForest":

                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators.get(),
                        max_depth=max_depth_value,
                        class_weight="balanced",
                    )
                    model.fit(X_train, y_train)

                elif model_type == "SVM":

                    model = SVC(
                        probability=True,
                        C=1,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                    )
                    model.fit(X_train, y_train)

                elif model_type == "KNN":

                    model = KNeighborsClassifier(
                        n_neighbors=self.knn_n_neighbors.get(),
                        weights=self.knn_weights.get(),
                        algorithm=self.knn_algorithm.get(),
                        p=self.knn_p.get(),
                        leaf_size=self.knn_leaf_size.get(),
                    )
                    model.fit(X_train, y_train)

                elif model_type == "Rede Neural":

                    try:
                        hidden_str = self.nn_hidden_layer.get().strip()
                        hidden_layer = tuple(int(x.strip()) for x in hidden_str.split(',') if x.strip() != "")
                        if not hidden_layer:
                            raise ValueError("A estrutura da camada oculta está vazia.")
                        max_iter_val = int(self.nn_max_iter.get())
                        lr_val = float(self.nn_lr.get())
                        solver_val = self.nn_solver.get()
                        momentum_val = float(self.nn_momentum.get()) if self.nn_momentum.get() else 0.9
                    except ValueError:
                        messagebox.showerror("Erro - Rede Neural",
                                             "Preencha todos os parâmetros da Rede Neural corretamente.")
                        self.after(0, self.stop_text_spinner)
                        return

                    if not (0 <= momentum_val <= 1):
                        messagebox.showerror("Erro - Rede Neural","O valor do momentum deve estar entre 0 e 1.")

                        self.after(0, self.stop_text_spinner)

                        return
                    # Prepara os parâmetros

                    params = dict(

                        hidden_layer_sizes=hidden_layer,
                        max_iter=max_iter_val,
                        random_state=50,
                        early_stopping=True,
                        learning_rate_init=lr_val,
                        solver=solver_val,

                    )

                    if solver_val == 'sgd':
                        params['momentum'] = momentum_val

                    model = MLPClassifier(**params)

                    model.fit(X_train, y_train)

                else:

                    messagebox.showerror("Erro - Modelo", f"Modelo '{model_type}' não reconhecido.")

                    self.after(0, self.stop_text_spinner)

                    return

                self.model = model  # Armazena o modelo treinado manualmente

            # ---------------- Verifica se CV está habilitado ----------------

            if self.use_cv_var.get():
                metric = self.cv_metric_var.get()
                folds = self.cv_k_folds.get()
                scores = cross_val_score(self.model, X_train, y_train, cv=folds, scoring=metric, n_jobs=-1)
                self.cv_scores = scores
                media = scores.mean()

            # ---------------- Métricas no teste holdout ----------------

            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)

            # Guarda para reuso ao alternar exibições (CV ↔ Confusão/ROC)
            self.acc, self.f1, self.cm, self.y_pred = acc, f1, cm, y_pred

            # Atualiza UI
            self.after(0, lambda: self.update_results(acc, f1, cm, y_test, y_pred, self.model))

            if acc >= 0.7:
                self.train_status.config(text="Máquina Treinada", foreground="#036B05")
            else:
                self.train_status.config(text="Máquina Não Treinada", foreground="#E80E0E")

            self.after(0, self.stop_text_spinner)

        except Exception as e:
            self.after(0, self.stop_text_spinner)
            mensagem_amigavel = self.traduzir_erro(str(e))
            messagebox.showerror("Erro no treino", mensagem_amigavel)

    def on_cv_checkbox_change(self, *args) -> None:
        """
        Dispara quando o checkbox de Validação Cruzada é marcado/desmarcado.
        Desmarca -> volta a exibir Matriz de Confusão e ROC com as métricas recentes (se existirem).
        """
        if not self.use_cv_var.get():
            # Voltando para Confusão/ROC (se já houver métricas calculadas)
            if all(getattr(self, attr, None) is not None for attr in ["acc", "f1", "cm", "y_test", "y_pred", "model"]):
                try:
                    self.update_results(self.acc, self.f1, self.cm, self.y_test, self.y_pred, self.model)
                except Exception as e:
                    messagebox.showerror("Erro", f"Ainda não há resultados para plotar: {e}")

            # Desabilitar os widgets de Validação Cruzada
            self.cv_metric_cb.config(state="disabled")
            self.cv_folds_spin.config(state="disabled")

        else:
            # Habilitar os widgets de Validação Cruzada
            self.cv_metric_cb.config(state="normal")
            self.cv_folds_spin.config(state="normal")

    def update_results(self, acc, f1, cm, y_test, y_pred, model) -> None:
        """
        Atualiza labels e plota:
          - Se CV ativo: Scores por fold + Boxplot da distribuição
          - Se CV inativo: Matriz de Confusão + Curva ROC (binária)
        """
        # Guarda as métricas atuais (para alternar telas sem recalcular)
        self.acc, self.f1, self.cm, self.y_pred = acc, f1, cm, y_pred

        # Acurácia
        self.acc_label.config(
            text=f"Acurácia: {acc * 100:.2f}%",
            foreground=("green" if acc >= 0.7 else "red")
        )

        # F1 macro
        self.f1_label.config(text=f"F1-score (macro): {f1:.4f}")

        # Texto explicativo da matriz
        self.cm_label.config(text="Matriz de Confusão (Linhas = Verdadeiro, Colunas = Predito)")

        # ---------------------- Caso: Validação Cruzada ----------------------
        if self.use_cv_var.get() and self.cv_scores is not None:
            for ax in self.axs:
                ax.clear()

            scores = self.cv_scores
            ax0, ax1 = self.axs

            # Gráfico 1: Barras dos folds com anotações
            bar_positions = range(1, len(scores) + 1)
            bars = ax0.bar(bar_positions, scores, color="#2ecc71", width=0.7)
            ax0.set_title("Scores por Fold", fontsize=10)
            ax0.set_xlabel("Fold", fontsize=9)
            ax0.set_ylabel(self.cv_metric_var.get(), fontsize=9)
            ax0.set_ylim(0, 1)
            ax0.set_xlim(0.5, len(scores) + 0.5)
            ax0.set_xticks(bar_positions)
            ax0.set_aspect("auto")
            ax0.margins(x=0.05, y=0.05)

            # Adiciona texto (% de desempenho) acima de cada barra
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax0.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,  # <- Posiciona no meio da barra
                    f"{score * 100:.2f}%",  # converte para porcentagem
                    ha="center", va="bottom", fontsize=8, fontweight="bold"
                )

            # Gráfico 2: Boxplot dos scores
            ax1.boxplot(scores, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#2ecc71", color="#27ae60"),
                        medianprops=dict(color="black"))
            ax1.set_title("Distribuição", fontsize=10)
            ax1.set_xticks([1])
            ax1.set_xticklabels([self.cv_metric_var.get()], fontsize=9)
            ax1.set_ylim(0, 1)
            ax1.yaxis.grid(True)

            # OU COLOCAR UM HISOTGRAMA

            # Gráfico 2: Histograma dos scores
            # ax1.hist(scores, bins=10, color="#2ecc71", edgecolor="#27ae60")
            # ax1.set_title("Distribuição dos Scores", fontsize=10)
            # ax1.set_xlabel(self.cv_metric_var.get(), fontsize=9)
            # ax1.set_ylabel("Frequência", fontsize=9)
            # ax1.set_xlim(0, 1)
            # ax1.grid(True)

            self.fig.tight_layout()
            self.canvas.draw()
            self.canvas.get_tk_widget().update_idletasks()
            return

        # ---------------------- Caso: sem Validação Cruzada ------------------
        for ax in self.axs:
            ax.clear()

        # Matriz de Confusão (binária)
        ax0 = self.axs[0]
        ax0.imshow(cm, interpolation="nearest", cmap="Greys")
        ax0.set_title("Métricas Matriciais")
        ax0.set_xticks([0, 1])
        ax0.set_yticks([0, 1])
        ax0.set_xticklabels(["Nao", "Sim"])
        ax0.set_yticklabels(["Nao", "Sim"])

        # Coloca os valores + rótulo em cada célula
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                if i == j:
                    color = "green"
                    label = "V.N" if i == 0 else "V.P"
                else:
                    if i == 0 and j == 1:
                        color = "red"
                        label = "F.P"
                    else:
                        color = "red"
                        label = "F.N"
                ax0.text(j, i, f"{cm[i][j]}\n{label}", ha="center", va="center", color=color, fontweight="bold")

        # Curva ROC (binária)
        ax1 = self.axs[1]
        if len(np.unique(y_test)) == 2:
            try:
                proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, label=f"Previsao (AUC = {roc_auc:.2f})", color="#13e80c")
                ax1.plot([0, 1], [0, 1], "r--", label="Chute")
                ax1.set_title("Curva ROC")
                ax1.set_xlabel("F.P")
                ax1.set_ylabel("")
                ax1.legend(loc="lower center")
            except Exception:
                ax1.text(0.1, 0.5, "")
        else:
            ax1.text(0.1, 0.5, "ROC apenas para classificação binária")

        self.canvas.draw()

    # =============================================================================
    # 13) PREDICAO MANUAL (UI)
    # =============================================================================
    def predicao_manual(self) -> None:
        """Janela para digitar valores das features e obter uma predição do modelo atual."""
        if self.df is None:
            messagebox.showerror("Erro - DataSet", "Faça o upload de um DataSet.")
            return

        if not hasattr(self, "model") or self.model is None:
            messagebox.showerror("Erro - Treinamento", "treine antes de realizar a predição manual!")
            return

        top = tk.Toplevel(self)
        top.title("Predição Manual")
        top.resizable(False, True)
        top.geometry("565x545")
        top.iconbitmap("machine_learning.ico")

        # Guarda referência da janela
        self.prediction_window = top

        # Features numéricas com base no df atual e alvo selecionado
        target = self.target_cb.get()
        features = self.df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()
        entries: dict[str, ttk.Entry] = {}

        ttk.Label(top, text="Insira os valores para as features:", font=("Arial", 12, "bold")).pack(pady=10)
        frame = ttk.Frame(top)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        for f in features:
            ttk.Label(frame, text=f).pack(anchor=tk.W)
            ent = ttk.Entry(frame)
            ent.pack(fill=tk.X, pady=2)
            entries[f] = ent

        result_label = ttk.Label(
            top, text="", font=("Arial", 12), foreground="green",
            wraplength=560, justify="center"
        )
        result_label.pack(pady=10)

        def predict_and_show() -> None:
            """Coleta entradas, aplica scaler (se houver) e exibe a predição."""
            result_label.config(text="", foreground="green")
            try:
                vals = []
                for f in features:
                    val = entries[f].get().strip()
                    if val == "":
                        raise ValueError(f"Preencha o campo  '{f}'")
                    vals.append(float(val))

                arr = np.array(vals).reshape(1, -1)
                if self.scaler is not None:
                    arr = self.scaler.transform(arr)

                pred = self.model.predict(arr)[0]
                if pred == 1:
                    result_label.config(
                        text=f"Você possui {target}. Classe 1",
                        foreground="red", font=("Arial", 12, "bold")
                    )
                else:
                    result_label.config(
                        text=f"Você não possui {target}. Classe 0",
                        foreground="green", font=("Arial", 12, "bold")
                    )

            except ValueError as ve:
                result_label.config(
                    text=(
                        f"Entrada inválida: {ve}. Certifique-se de preencher todos os campos com números válidos."
                    ),
                    foreground="red", font=("Arial", 12, "bold")
                )
            except Exception as e:
                result_label.config(
                    text=(
                        "Erro inesperado durante a predição.\n"
                        "Verifique se:\n"
                        "- Todos os campos estão preenchidos com números.\n"
                        "- Os dados inseridos seguem o formato esperado.\n"
                        "- O modelo foi treinado corretamente.\n"
                        f"Detalhes técnicos: {e}"
                    ),
                    foreground="red", font=("Arial", 12, "bold")
                )

        btn_prever = tk.Button(
            top, text="Prever", bg="#008B8B", fg="white",
            activebackground="#20B2AA", relief="flat",
            font=("Arial", 11, "bold"), width=12,
            command=predict_and_show
        )
        btn_prever.pack(pady=8)
        btn_prever.bind("<Enter>", lambda e: btn_prever.config(bg="#20B2AA"))
        btn_prever.bind("<Leave>", lambda e: btn_prever.config(bg="#008B8B"))

    # =============================================================================
    # 14) PONTO DE ENTRADA DO APP
    # =============================================================================


# Execução (mantida)
if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
