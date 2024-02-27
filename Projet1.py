import streamlit as st
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import unittest


data = pd.read_csv(r'Autosphere_data.csv',sep=';')

data['Carburant'].replace({'Essence M': 'Essence', 'ESSENCE': 'Essence','Essence B': 'Essence','HYBRIDE': 'Hybride','Hybride R': 'Hybride','Electrique': 'Hybride','Diesel M': 'Diesel','DIESEL': 'Diesel','Gpl':'Gaz'}, inplace=True)
data['Type_moteur'].replace({'460ch':'V12','8v':'V8','116ch':'V8','T':'TSI'},inplace=True)
data['Transmission'].replace({'AUTOMATIQUE':'Automatique','MANUELLE':'Manuelle'},inplace=True)
class Visualisation:
    def __init__(self, file_path):
        try:
            self.df = pd.read_csv(file_path, delimiter=';')
        except FileNotFoundError:
            print(f"File {file_path} not found.")

    def display_head(self, n=5):
        return self.df.head(n)
    
    def display_data_types(self):
        return self.df.dtypes
    
    def plot_scatter_km_prix(self):
        sns.scatterplot(x='KM', y='Prix', data=self.df).set_title("Prix des voitures en fonction du kilométrage")
        plt.show()

    def barplot_annee_prix(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Année', y='Prix', data=self.df, errorbar=None)
        plt.title("Évolution du prix moyen en fonction de l'année")
        plt.xlabel('Année')
        plt.ylabel('Prix moyen')
        plt.show()

    def boxplot_par_marques(self):
        plt.figure(figsize=(12, 8))
        self.df.boxplot(column='Prix', by='Marque', vert=False, figsize=(10, 8))
        plt.title('Répartition des prix par marque')
        plt.xlabel('Prix')
        plt.ylabel('Marque')
        plt.tight_layout()
        plt.show()

    def display_correlation(self):
        correlation_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Matrice de corrélation")
        plt.show()
class PCAStatisticsCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_pca_statistics(self):
        try:
            sc = StandardScaler()
            types = self.df.dtypes
            colonnes_strings = types[types == object]
            colonnes_a_supprimer = colonnes_strings.index.tolist()
            df_sans_strings = self.df.drop(columns=colonnes_a_supprimer)

            n = df_sans_strings.shape[0]
            p = df_sans_strings.shape[1]

            Z = sc.fit_transform(df_sans_strings)
            acp = PCA(svd_solver='full')
            coord = acp.fit_transform(Z)

            eigval = (n-1)/n * acp.explained_variance_
            bs = 1 / np.arange(p, 0, -1)
            bs = np.cumsum(bs)
            bs = bs[::-1]

            di = np.sum(Z**2, axis=1)
            cos2 = coord**2 / di[:, None]

            ctr = coord**2 / (n * eigval)

            return eigval, bs, cos2, ctr

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

class TestPCAStatisticsCalculator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame()

    def test_calculate_pca_statistics(self):
        pca_calculator = PCAStatisticsCalculator(self.df)
        result = pca_calculator.calculate_pca_statistics()

        self.assertIsNotNone(result)
class DataProcessor:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Les données doivent être de type DataFrame")
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def separate_features_target(self):
        if 'Prix' not in self.data.columns:
            raise ValueError("La colonne 'Prix' est introuvable dans les données")
        self.X = self.data.drop('Prix', axis=1)
        self.y = self.data['Prix']
        return self.X, self.y

    def split_train_test(self):
        if self.X is None or self.y is None:
            raise ValueError("Les caractéristiques et la cible doivent être séparées en premier")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=5
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Les données d'entraînement et de test doivent être divisées en premier")
        numeric_features = ['Chevaux', 'Cylindrée', 'CO2', 'KM', 'Année', 'Puissance', 'Puissance_fiscale']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_features = ['Marque', 'Modèle', 'Carburant', 'Type_moteur', 'Transmission']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        return self.X_train, self.X_test

class PCAAnalyzer:
  try:
      def __init__(self, df):
          self.df = df

      def preprocess_data(self):
          types = self.df.dtypes
          colonnes_strings = types[types == object]
          colonnes_a_supprimer = colonnes_strings.index.tolist()
          self.df_sans_strings = self.df.drop(columns=colonnes_a_supprimer)

      def perform_pca(self):
          sc = StandardScaler()
          n = self.df_sans_strings.shape[0]
          p = self.df_sans_strings.shape[1]
          Z = sc.fit_transform(self.df_sans_strings)
          acp = PCA(svd_solver='full')
          coord = acp.fit_transform(Z)
          eigval = (n - 1) / n * acp.explained_variance_
          sqrt_eigval = np.sqrt(eigval)
          self.sqrt_eigval = sqrt_eigval
          self.coord = coord
          self.eigval = eigval
          self.acp = acp
  except Exception as e:
            print(f"An error occurred during PCA: {e}")
class ModelTrainerEvaluator:
    def __init__(self, model, X_test, y_test):
        if model is None or X_test is None or y_test is None:
            raise ValueError("Le modèle ou les données de test ne peuvent pas être vides")
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        return mse, r2

    def display_coefficients_and_intercept(self):
        if not hasattr(self.model, 'coef_') or not hasattr(self.model, 'intercept_'):
            raise AttributeError("Le modèle ne possède pas les attributs de coefficient et d'intercept")
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        return coefficients, intercept

    def plot_residuals(self):
        predictions = self.model.predict(self.X_test)
        residuals = self.y_test - predictions
        plt.scatter(predictions, residuals)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.show()

    def test_evaluate_model(self):
        # Test unitaire pour évaluer_model
        test_predictions = np.array([1, 2, 3, 4])
        test_y_test = np.array([2, 3, 4, 5])
        dummy_model = lambda x: x
        test_mse, test_r2 = self.evaluate_model(dummy_model, test_predictions, test_y_test)
        assert test_mse == 1.0, "Erreur dans le calcul de la MSE"
        assert test_r2 == 0.0, "Erreur dans le calcul de R²"

    def test_display_coefficients_and_intercept(self):
        # Test unitaire pour display_coefficients_and_intercept
        dummy_model = lambda x: x
        dummy_model.coef_ = np.array([0.5, 0.7])
        dummy_model.intercept_ = 1.0
        test_coefficients, test_intercept = self.display_coefficients_and_intercept(dummy_model)
        assert np.array_equal(test_coefficients, np.array([0.5, 0.7])), "Coefficients incorrects"
        assert test_intercept == 1.0, "Intercept incorrect"
class PCAResultsVisualizer:
    def __init__(self, pca_analyzer):
        self.pca_analyzer = pca_analyzer

    def plot_scree_plot(self):
        plt.plot(np.arange(1, len(self.pca_analyzer.eigval) + 1), self.pca_analyzer.eigval)
        plt.title("Scree plot")
        plt.ylabel("Eigen values")
        plt.xlabel("Factor number")
        plt.show()

    def plot_cumulative_explained_variance(self):
        cumsum_explained_variance = np.cumsum(self.pca_analyzer.acp.explained_variance_ratio_)
        plt.plot(np.arange(1, len(cumsum_explained_variance) + 1), cumsum_explained_variance)
        plt.title("Explained variance vs. number of factors")
        plt.ylabel("Cumsum explained variance ratio")
        plt.xlabel("Factor number")
        plt.show()

    def plot_individuals_map(self):
        sqrt_eigval = self.pca_analyzer.sqrt_eigval
        coord = self.pca_analyzer.coord
        fig, axes = plt.subplots(figsize=(12, 12))
        axes.set_xlim(-6, 6)
        axes.set_ylim(-6, 6)
        plt.scatter(coord[:, 0], coord[:, 1])
        plt.plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
        plt.title("Carte des individus (PC1 vs PC2)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()
        plt.show()


    def plot_correlation_circle(self):
      sqrt_eigval = self.pca_analyzer.sqrt_eigval
      acp = self.pca_analyzer.acp
      corvar = np.zeros((acp.components_.shape[1], acp.components_.shape[1]))

      for k in range(acp.components_.shape[1]):
          corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]

      fig, axes = plt.subplots(figsize=(8, 8))
      axes.set_xlim(-1, 1)
      axes.set_ylim(-1, 1)

      for j in range(corvar.shape[1]):
          plt.annotate(self.pca_analyzer.df_sans_strings.columns[j], (corvar[j, 0], corvar[j, 1]))

          plt.arrow(0, 0, corvar[j, 0], corvar[j, 1], head_width=0.05, head_length=0.1, fc='orange', ec='orange')

      plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
      plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

      cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
      axes.add_artist(cercle)

      plt.title("Cercle des corrélations avec vecteurs")
      plt.xlabel("PC1")
      plt.ylabel("PC2")
      plt.show()
class PCAGraphsVisualizer:
    def __init__(self, eigval, bs, cos2, ctr):
        self.eigval = eigval
        self.bs = bs
        self.cos2 = cos2
        self.ctr = ctr

    def plot_eigenvalues_and_thresholds(self):
        try:
            p = len(self.eigval)
            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(1, p + 1), self.eigval, color='skyblue')
            plt.plot(np.arange(1, p + 1), self.bs, color='orange', marker='o', linestyle='-')
            plt.xlabel('Nombre de Facteurs')
            plt.ylabel('Valeurs Propres')
            plt.title('Valeurs propres et Seuils')
            plt.legend(['Seuils', 'Valeurs Propres'])
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_cos2(self):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.cos2[:, 0], self.cos2[:, 1], color='green')
            plt.xlabel('COS2 sur F1')
            plt.ylabel('COS2 sur F2')
            plt.title('COS2 sur les deux premiers axes factoriels')
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_ctr(self):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.ctr[:, 0], self.ctr[:, 1], color='red')
            plt.xlabel('CTR sur F1')
            plt.ylabel('CTR sur F2')
            plt.title('CTR sur les deux premiers axes factoriels')
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")
class TestPCAGraphsVisualizer(unittest.TestCase):
    def setUp(self):
        self.eigval = np.array([1, 2, 3])  # Simuler des données pour les tests
        self.bs = np.array([0.5, 1.5, 2.5])  # Simuler des données pour les tests
        self.cos2 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Simuler des données pour les tests
        self.ctr = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])  # Simuler des données pour les tests

    def test_plot_eigenvalues_and_thresholds(self):
        graphs_visualizer = PCAGraphsVisualizer(self.eigval, self.bs, self.cos2, self.ctr)
        graphs_visualizer.plot_eigenvalues_and_thresholds()
class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Carburant': ['Essence M', 'Diesel M', 'AUTOMATIQUE'],
            'Type_moteur': ['460ch', '8v', 'T'],
            'Transmission': ['AUTOMATIQUE', 'MANUELLE', 'AUTOMATIQUE']
        })

    def test_clean_carburant(self):
        clean_carburant(self.data)
        expected_result = ['Essence', 'Diesel', 'AUTOMATIQUE']
        self.assertListEqual(list(self.data['Carburant']), expected_result)

    def test_clean_type_moteur(self):
        clean_type_moteur(self.data)
        expected_result = ['V12', 'V8', 'TSI']
        self.assertListEqual(list(self.data['Type_moteur']), expected_result)

    def test_clean_transmission(self):
        clean_transmission(self.data)
        expected_result = ['Automatique', 'Manuelle', 'Automatique']
        self.assertListEqual(list(self.data['Transmission']), expected_result)



class ChevauxProcc:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Les données doivent être de type DataFrame")
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def separate_features_target_ch(self):
        if 'Chevaux' not in self.data.columns:
            raise ValueError("La colonne 'Chevaux' est introuvable dans les données")
        self.X = self.data.drop('Chevaux', axis=1)
        self.y = self.data['Chevaux']
        return self.X, self.y

    def split_train_test_ch(self):
        if self.X is None or self.y is None:
            raise ValueError("Les caractéristiques et la cible doivent être séparées en premier")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=5
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data_ch(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Les données d'entraînement et de test doivent être divisées en premier")
        numeric_features = [ 'Cylindrée', 'Puissance_fiscale']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_features = ['Marque', 'Modèle', 'Carburant', 'Type_moteur', 'Transmission']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        return self.X_train, self.X_test
def clean_carburant(data):
    data['Carburant'].replace({'Essence M': 'Essence', 'ESSENCE': 'Essence','Essence B': 'Essence','HYBRIDE': 'Hybride','Hybride R': 'Hybride','Electrique': 'Hybride','Diesel M': 'Diesel','DIESEL': 'Diesel','Gpl':'Gaz'}, inplace=True)
    print('\n Carburant : \n')
    return data['Carburant'].unique()
def clean_type_moteur(data):
    data['Type_moteur'].replace({'460ch':'V12','8v':'V8','116ch':'V8','T':'TSI'},inplace=True)
    print('\n Type de moteur : \n')
    return data['Type_moteur'].unique()
def clean_transmission(data):
    data['Transmission'].replace({'AUTOMATIQUE':'Automatique','MANUELLE':'Manuelle'},inplace=True)
    print('\n Transmission : \n')
    return data['Transmission'].unique()

data_processor = DataProcessor(data)
X, y = data_processor.separate_features_target()
X_train, X_test, y_train, y_test = data_processor.split_train_test()
X_train_processed, X_test_processed = data_processor.preprocess_data()

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_processed, y_train)

evaluator = ModelTrainerEvaluator(linear_regression_model, X_test_processed, y_test)
evaluator.evaluate_model()
evaluator.display_coefficients_and_intercept()
evaluator.plot_residuals()

ch_processor = ChevauxProcc(data)
X_ch, y_ch = ch_processor.separate_features_target_ch()
X_train_ch, X_test_ch, y_train_ch, y_test_ch = ch_processor.split_train_test_ch()
X_train_processed_ch, X_test_processed_ch = ch_processor.preprocess_data_ch()

linear_regression_model_ch = LinearRegression()
linear_regression_model_ch.fit(X_train_processed_ch, y_train_ch)

evaluator_ch = ModelTrainerEvaluator(linear_regression_model_ch, X_test_processed_ch, y_test_ch)
evaluator_ch.evaluate_model()
evaluator_ch.display_coefficients_and_intercept()
evaluator_ch.plot_residuals()
new_data_ch =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant','Type_moteur','Cylindrée','Année','Transmission','Puissance_fiscale'])
new_data =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant', 'Chevaux','Type_moteur','Cylindrée','CO2','KM','Année','Transmission','Puissance','Puissance_fiscale'])
def estimer_ch(Marque, Model, Carburant,Type_moteur,Cylindree,Annee,Transmission,Puissance_fiscale):
    new_data_ch.loc['0'] = [Marque, Model, Carburant,Type_moteur,Cylindree,Annee,Transmission,Puissance_fiscale]
    new_data_processed = ch_processor.preprocessor.transform(new_data_ch)
    predictions_new_data = linear_regression_model_ch.predict(new_data_processed)
    return int(predictions_new_data[0])
def estimer_prx(Marque, Model, Carburant, Chevaux,Type_moteur,Cylindree,CO2,Kilometrage,Annee,Transmission,Puissance,Puissance_fiscale):
    new_data.loc['0'] = [Marque, Model, Carburant, Chevaux,Type_moteur,Cylindree,CO2,Kilometrage,Annee,Transmission,Puissance,Puissance_fiscale]
    new_data_processed = data_processor.preprocessor.transform(new_data)
    predictions_new_data = linear_regression_model.predict(new_data_processed)
    return(int(predictions_new_data[0]*3.34))
def main():
    st.title("Estimateur de prix d'une voiture")
    menu = ["Estimer le prix de votre voiture", "Estimer le Prix d'une voiture sur Tayara","Visualiser le code"]
    with st.sidebar :
        choice = option_menu(menu_title = "Menu",options=menu,icons=["piggy-bank-fill","airplane-fill","code-square"])
    Carb = ["Essence","Diesel","Hybride","Gaz"]
    moteur=['PureTech', 'BlueHDi', 'TCe', 'Blue' ,'Flexifuel' ,'EcoBoost' ,'EcoBlue', 'dCi' ,'TSI' ,'EcoTSI' ,'TDI','V6' ,'V8' ,'V12']
    trans=['Manuelle','Automatique']
    if choice=="Estimer le prix de votre voiture":
        st.subheader("Estimer le prix de votre voiture")
        with st.form(key="esti_voit"):
            col1, col2 = st.columns([3, 1])
            with col1:
                Marque = st.text_input("Marque")
                Model=st.text_input("Modèle")
                Annee=int(st.number_input("Année",format='%d',min_value=0,max_value=2023))
                Carburant=st.selectbox("Carburant",Carb)
                Transmission=st.selectbox("Transmission",trans)
                Puissance_fiscale=int(st.number_input("Puissance Fiscale",format='%d',min_value=0))
                Type_moteur=st.selectbox("Type de moteur",moteur)
                Kilometrage=int(st.number_input("Kilométrage",format='%d',min_value=1000))
                Cylindree=st.number_input("Cylindrée (en L)",min_value=1.0,max_value=10.0)
                Chevaux=int(st.number_input("Puissance (en chevaux)",format='%d',min_value=0))
                Puissance=Chevaux*0.7355
                CO2=int((Puissance_fiscale-(Puissance/40)*1.6)*45)
            with col2:
                for i in range(49):
                    st.write("")
                button_clicked = st.form_submit_button("Approximer")
            submit_button = st.form_submit_button(label='Estimer')
            if button_clicked:
                st.success("Votre voiture fait environ : {} ch".format(estimer_ch(Marque, Model, Carburant,Type_moteur,Cylindree,Annee,Transmission,Puissance_fiscale)))



        if submit_button:
            st.success("Votre voiture vaut environ : {} dt".format(estimer_prx(Marque, Model, Carburant, Chevaux,Type_moteur,Cylindree,CO2,Kilometrage,Annee,Transmission,Puissance,Puissance_fiscale)))

    elif choice =="Estimer le Prix d'une voiture sur Tayara":
        st.subheader("Estimer le Prix d'une voiture sur Tayara")
        with st.form(key="esti_tay"):
            url = st.text_input("URL Tayara.tn")
            submit_button2 = st.form_submit_button(label='Estimer')
        if submit_button2:
            if ("tayara.tn" in url and "Voitures" in url):
                spec=[]
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                l_specs=soup.find_all('span',class_="text-gray-700/80 text-xs md:text-sm lg:text-sm font-semibold")
                for i in range(len(l_specs)):
                    spec.append(l_specs[i].text)
                kilometrage=spec[0]
                Transmission=spec[3]
                Annee=int(spec[4])
                Cylindree=float(re.sub(r"[a-zA-Z]", "", spec[5]))
                Marque=spec[6]
                Modele=spec[7]
                Chevaux_fisc=int(spec[8])
                Carburant=spec[10]
                Type=spec[9]
                Chevaux=estimer_ch(Marque,Modele,Carburant,Type,Cylindree,Annee,Transmission,Chevaux_fisc)
                Puissance=Chevaux*0.7355
                CO2=int((Chevaux_fisc-(Puissance/40)*1.6)*45)
                st.success("Votre voiture vaut environ : {} dt".format(estimer_prx(Marque,Modele,Carburant,Chevaux,Type,Cylindree,CO2,kilometrage,Annee,Transmission,Puissance,Chevaux_fisc)))
            else :
                st.error("Vous devez saisir un lien du site tayara.tn de la catégorie Voitures ")
        

    else:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        choice2 = option_menu(menu_title = None,options=["Web Scraping Et Visualisation","Modelisation","GUI"],icons=["browser-chrome","clipboard2-data-fill","pc-display-horizontal"],orientation="horizontal")
        if choice2=="Web Scraping Et Visualisation":
            st.markdown("## **1.Importation des bibliothèques et création d'un décorateur pour mesurer le temps d'exécution des fonctions**")
            st.markdown("""```sql 
                        import requests
            from bs4 import BeautifulSoup
            import csv
            import pandas as pd
            import re
            from functools import wraps
            import time""")
            st.markdown("""```sql 
                        def calculate_time_spent(function):
    #Calcule le temps que met une fonction à s'exécuter.
    @wraps(function)
    def wrapper(*args, **kwargs):
        #Décore la fonction.
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        time_spent = end - start
        print(f"Secondes passées: {time_spent:.2f}")
        return result
    return wrapper""")
            st.markdown("""## **2.Web scraping de la page autosphere**

Nous avons créé cette classe pour **scraper les données** du site web AutoSphere.Elle a une méthode scrape_data qui contient le scraping, elle effectue les actions suivantes :

-Parcourir les pages une par une (il y a 540 pages).

-Utiliser **BeautifulSoup** pour extraire les données que nous avons besoins(marque, modèle, prix, caractéristiques du moteur, etc...) en utilisant les **expressions régulières**.

-Mettre toutes les données dans un **dataFrame** pandas.

De plus, on a utilisé **le try-except** pour capturer toute exception qui pourrait se produire lors du scraping
""")
            st.markdown("""```sql 
                        class AutoSphereScraper:
    def __init__(self):
        self.df = pd.DataFrame(columns=['Marque', 'Modèle', 'Prix', 'Carburant', 'Chevaux', 'Type_moteur', 'Cylindrée', 'CO2', 'KM', 'Année', 'Transmission', 'Puissance', 'Puissance_fiscale'])
    @calculate_time_spent
    def scrape_data(self):
      try:
          base_url = "https://www.autosphere.fr/recherche?market=VO&page={}&critaire_checked[]=year&critaire_checked[]=discount&critaire_checked[]=emission_co2"
          total_pages = 540

          for page_number in range(1, total_pages + 1):
              url = base_url.format(page_number)
              response = requests.get(url)
              soup = BeautifulSoup(response.content, 'html.parser')

              serie_element = soup.find_all('span', class_='serie ellipsis')
              cars = soup.find_all('span', class_='designation_enfant')
              liste_prix = soup.select('span.bloc_prix, span.bloc_prix.fvo-color')
              caracteristiques = soup.find_all('div', class_='caract')

              data_to_append = []

              for index, car in enumerate(cars):
                  marque = car.find('span', class_='marque').text.strip()
                  modele = car.find('span', class_='modele').text.strip()
                  serie_text = serie_element[index].get_text(strip=True)
                  prix_text = liste_prix[index].get_text(strip=True)
                  chiffres_prix = ''.join(filter(str.isdigit, prix_text))

                  cylindree = ''
                  type_moteur = ''
                  chevaux = ''
                  puissance = ''
                  puissance_fiscale = None

                  if serie_text:
                      serie_info = serie_text.split(' ')
                      if len(serie_info) >= 4:
                          match_cylindre_type = re.search(r'(\d+\.\d+)\s+(\w+)', serie_text)
                          cylindree = match_cylindre_type.group(1) if match_cylindre_type else None
                          type_moteur = match_cylindre_type.group(2) if match_cylindre_type else None
                          match_chevaux = re.search(r'(\d+(\.\d+)?)ch', serie_text)
                          chevaux = match_chevaux.group(1) if match_chevaux else None
                          match_co2 = re.search(r'(\d+)g\b', serie_text, re.IGNORECASE)
                          CO2 = match_co2.group(1) if match_co2 else None
                          match_puissance = re.search(r'(\d+(\.\d+)?)\s*(ch|kw)', serie_text, re.IGNORECASE)
                          puissance = match_puissance.group(1) if match_puissance else None
                          if CO2 and puissance:
                              if 'ch' in serie_text.lower():
                                  puissance = float(puissance) * 0.7355
                              puissance_fiscale = (int(CO2) / 45) + (float(puissance) / 40) * 1.6
                  caract = [mot.strip() for mot in re.findall(r'[^/\n]+', caracteristiques[index].text) if mot.strip()]
                  if len(caract) >= 4:
                      carburant = caract[0]
                      chiffres_km = ''.join(filter(str.isdigit, caract[1]))
                      annee = caract[2]
                      transmission = caract[3]
                  data_to_append.append({
                      'Marque': marque,
                      'Modèle': modele,
                      'Prix': chiffres_prix,
                      'Carburant': carburant,
                      'Cylindrée': cylindree,
                      'Type_moteur': type_moteur,
                      'Chevaux': chevaux,
                      'CO2': CO2,
                      'KM': chiffres_km,
                      'Année': annee,
                      'Transmission': transmission,
                      'Puissance': puissance,
                      'Puissance_fiscale': puissance_fiscale
                  })

              self.df = pd.concat([self.df, pd.DataFrame(data_to_append)], ignore_index=True)

          print("Scrapping done!")
          return self.df
      except Exception as e:
          print(f"An error occurred: {e}")
          return None""")
            st.markdown("""## **3-Exécution et téléchargement**

Nous avons maintenant utilisé **notre décorateur** pour mesurer le temps d'exécution des deux fonctions :

save_to_csv : qui enregistre nos données dans un fichier csv.

execute_web_scraping : qui crée une instance de AutoSphereScraper et qui exécute la méthode scrape_data.
""")
            st.markdown(""" ```sql @calculate_time_spent
def save_to_csv(data, filename='autosphere_data.csv'):
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

@calculate_time_spent
def execute_web_scraping():
    scraper = AutoSphereScraper()
    data = scraper.scrape_data()
    return data""")
            st.markdown("""**-Exécution**""")
            st.markdown("""```sql 
                        scraped_data = execute_web_scraping()
save_to_csv(scraped_data)""")
            st.markdown("""## **4-Analyse visuelle des données**

Nous avons créé la classe **Visualisation** ,au début nous avons ajouter **une exception** pour voir si le fichier csv existe puis on les a chargées dans un **DataFrame** pandas.
Ensuite nous avons créer 4 méthodes de visualisation :

-Kilométrage et Prix (nuage de points).

-Année et prix (barplot).

-Marque et prix (Boxplot).

-Matrice de corrélation (heatmap)
""")
            st.markdown("""```sql 
                        class Visualisation:
    def __init__(self, file_path):
        try:
            self.df = pd.read_csv(file_path, delimiter=';')
        except FileNotFoundError:
            print(f"File {file_path} not found.")

    def display_head(self, n=5):
        return self.df.head(n)
    def display_data_types(self):
        return self.df.dtypes
    def plot_scatter_km_prix(self):
        sns.scatterplot(x='KM', y='Prix', data=self.df).set_title("Prix des voitures en fonction du kilométrage")
        plt.show()

    def barplot_annee_prix(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Année', y='Prix', data=self.df, errorbar=None)
        plt.title("Évolution du prix moyen en fonction de l'année")
        plt.xlabel('Année')
        plt.ylabel('Prix moyen')
        plt.show()
    def boxplot_par_marques(self):
        plt.figure(figsize=(12, 8))
        self.df.boxplot(column='Prix', by='Marque', vert=False, figsize=(10, 8))
        plt.title('Répartition des prix par marque')
        plt.xlabel('Prix')
        plt.ylabel('Marque')
        plt.tight_layout()
        plt.show()
    def display_correlation(self):
        correlation_matrix = self.df.corr(numeric_only=True)  # Spécifiez explicitement numeric_only
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Matrice de corrélation")
        plt.show()

file_path = "/content/drive/MyDrive/ColabNotebooks/autosphere_data.csv"
Visualisation = Visualisation(file_path)
print(' les premières lignes :  \n')
print(Visualisation.display_head())
print('\n les types des données des colonnes :  \n')
print(Visualisation.display_data_types())""")
            file_path = "Autosphere_data.csv"
            visualisation_instance = Visualisation(file_path)
            st.markdown("### ***Voici les premières données***")
            st.write(visualisation_instance.display_head())
            st.markdown("### ***Ainsi que leur type***") 
            st.write(visualisation_instance.display_data_types())  
            st.markdown("""## Affichage des graphiques

-KM et prix :
On peut observer que les prix se situent autour des 25000 malgré les différences de kilométrage mais il ya quelques **points abérants** qui s'écarte des autres données comme la voiture avec 60000KM et qui est aux alentours de 175000 sachant que plusieurs voitures ont fait le meme kilométrage et qui sont aux alentours de 25000
""") 
            st.markdown("""```sql 
                        # Affichage des graphiques
Visualisation.plot_scatter_km_prix()""")
            fig=visualisation_instance.plot_scatter_km_prix()
            st.pyplot(fig)
            st.markdown("""-Année et prix :
On peut observer que les prix varient entre 18000 et 26000 mais qu'il ya un point abérant pour l'année 2015 puisque ca augmente jusqu'a presque 35000
""")
            st.markdown("""```sql 
                        Visualisation.barplot_annee_prix()""")
            fig=visualisation_instance.barplot_annee_prix()
            st.pyplot(fig)
            st.markdown("""-Marque et prix :
On peut observer que les marques PEUGEOT et CITROEN ont le plus de points abérants et que la moyenne des prix sont autours de 25000 pour presque toute les marques
""")
            st.markdown("""```sql
                        Visualisation.boxplot_par_marques()""")
            fig=visualisation_instance.boxplot_par_marques()
            st.pyplot(fig)
            st.markdown("""-Matrice de corrélation :
on peut observer qu'il ya une **forte corrélation positive** entre chevaux et cylindrée ainsi avec chevaux et puissance_fiscale mais une **forte corrélation négative** entre Année et KM
""")
            st.markdown("""```sql
                        Visualisation.display_correlation()""")
            fig=visualisation_instance.display_correlation()
            st.pyplot(fig)
            st.markdown("""## **5-Analyse en composantes principales (ACP)**

Nous avons défini une classe PCAAnalyzer pour faire l'ACP, nous avons ajouté **une exception** au cas où il y une erreur au cours de l'ACP.
La méthode perform_pca normalise les données puis applique l'acp ensuite calcule les eigenvalues , les sqrt_eigval  et les coordinates.Tous les résultats sont stockés comme attributs de classe
""")
            st.markdown("""```sql
                        from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class PCAAnalyzer:
  try:
      def __init__(self, df):
          self.df = df

      def preprocess_data(self):
          types = self.df.dtypes
          colonnes_strings = types[types == object]
          colonnes_a_supprimer = colonnes_strings.index.tolist()
          self.df_sans_strings = self.df.drop(columns=colonnes_a_supprimer)

      def perform_pca(self):
          sc = StandardScaler()
          n = self.df_sans_strings.shape[0]
          p = self.df_sans_strings.shape[1]
          Z = sc.fit_transform(self.df_sans_strings)
          acp = PCA(svd_solver='full')
          coord = acp.fit_transform(Z)
          eigval = (n - 1) / n * acp.explained_variance_
          sqrt_eigval = np.sqrt(eigval)
          self.sqrt_eigval = sqrt_eigval
          self.coord = coord
          self.eigval = eigval
          self.acp = acp
  except Exception as e:
            print(f"An error occurred during PCA: {e}")""")
            st.markdown("""## **6-Visualisation des Résultats de l'ACP**

Nous avons défini une classe PCAResultsVisualizer qui permet de visualiser les résultats de l'ACP. il ya 4 méthodes:

-théorème du coude (Scree plot).

-variance cumulée par chaque facteur (plot).

-carte des individus.

-cercle des corrélation.
""")
            st.markdown("""```sql
                        class PCAResultsVisualizer:
    def __init__(self, pca_analyzer):
        self.pca_analyzer = pca_analyzer

    def plot_scree_plot(self):
        plt.plot(np.arange(1, len(self.pca_analyzer.eigval) + 1), self.pca_analyzer.eigval)
        plt.title("Scree plot")
        plt.ylabel("Eigen values")
        plt.xlabel("Factor number")
        plt.show()

    def plot_cumulative_explained_variance(self):
        cumsum_explained_variance = np.cumsum(self.pca_analyzer.acp.explained_variance_ratio_)
        plt.plot(np.arange(1, len(cumsum_explained_variance) + 1), cumsum_explained_variance)
        plt.title("Explained variance vs. number of factors")
        plt.ylabel("Cumsum explained variance ratio")
        plt.xlabel("Factor number")
        plt.show()

    def plot_individuals_map(self):
        sqrt_eigval = self.pca_analyzer.sqrt_eigval
        coord = self.pca_analyzer.coord
        fig, axes = plt.subplots(figsize=(12, 12))
        axes.set_xlim(-6, 6)
        axes.set_ylim(-6, 6)
        plt.scatter(coord[:, 0], coord[:, 1])
        plt.plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
        plt.title("Carte des individus (PC1 vs PC2)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()
        plt.show()


    def plot_correlation_circle(self):
      sqrt_eigval = self.pca_analyzer.sqrt_eigval
      acp = self.pca_analyzer.acp
      corvar = np.zeros((acp.components_.shape[1], acp.components_.shape[1]))

      for k in range(acp.components_.shape[1]):
          corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]

      fig, axes = plt.subplots(figsize=(8, 8))
      axes.set_xlim(-1, 1)
      axes.set_ylim(-1, 1)

      for j in range(corvar.shape[1]):
          plt.annotate(self.pca_analyzer.df_sans_strings.columns[j], (corvar[j, 0], corvar[j, 1]))

          plt.arrow(0, 0, corvar[j, 0], corvar[j, 1], head_width=0.05, head_length=0.1, fc='orange', ec='orange')

      plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
      plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

      cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
      axes.add_artist(cercle)

      plt.title("Cercle des corrélations avec vecteurs")
      plt.xlabel("PC1")
      plt.ylabel("PC2")
      plt.show()""")
            st.markdown("""**-Création de l'analyseur de l'ACP et création du visualiseur des résultats**""")
            st.markdown("""```sql
                        pca_analyzer = PCAAnalyzer(Visualisation.df)
pca_analyzer.preprocess_data()
pca_analyzer.perform_pca()
pca_visualizer = PCAResultsVisualizer(pca_analyzer)""")
            pca_analyzer = PCAAnalyzer(visualisation_instance.df)
            pca_analyzer.preprocess_data()
            pca_analyzer.perform_pca()
            pca_visualizer = PCAResultsVisualizer(pca_analyzer)
            st.markdown("""# **-Affichage des graphs**""")
            st.markdown("""```sql 
# Affichage du scree plot
pca_visualizer.plot_scree_plot()
""")
            fig=pca_visualizer.plot_scree_plot()
            st.pyplot(fig)
            st.markdown("""```sql
                        #affichage de la variance expliquée cumulée:
pca_visualizer.plot_cumulative_explained_variance()""")
            fig=pca_visualizer.plot_cumulative_explained_variance()
            st.pyplot(fig)
            st.markdown("""```sql
                        #affichage de la carte des individus :
pca_visualizer.plot_individuals_map()""")
            fig=pca_visualizer.plot_individuals_map()
            st.pyplot(fig)
            st.markdown("""Le prix montre une **faible corréation** avec les composantes principales.
KM a une **corrélation positive** avec PC2 et l'année a **une corrélation negative** avec PC2
les autres variable sont **corrélées positivement** avec PC1
""")
            st.markdown("""```sql                        
#affichage du cercle de coorelation
pca_visualizer.plot_correlation_circle()""")
            fig=pca_visualizer.plot_correlation_circle()
            st.pyplot(fig)
            st.markdown("""## **7- Calcul des statistiques de l'ACP**

Cette partie calcule et affiche les valeurs propres, les cos2 et les ctr en utilisant **numpy**.


**Les tests unitaires** associés à cette classe assurent que les méthodes de calcul de statistiques fonctionnent sans erreur. Ils vérifient également si les résultats produits sont non nuls et cohérents.
""")
            st.markdown("""```sql
                        class PCAStatisticsCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_pca_statistics(self):
        try:
            sc = StandardScaler()
            types = self.df.dtypes
            colonnes_strings = types[types == object]
            colonnes_a_supprimer = colonnes_strings.index.tolist()
            df_sans_strings = self.df.drop(columns=colonnes_a_supprimer)

            n = df_sans_strings.shape[0]
            p = df_sans_strings.shape[1]

            Z = sc.fit_transform(df_sans_strings)
            acp = PCA(svd_solver='full')
            coord = acp.fit_transform(Z)

            eigval = (n-1)/n * acp.explained_variance_
            bs = 1 / np.arange(p, 0, -1)
            bs = np.cumsum(bs)
            bs = bs[::-1]

            di = np.sum(Z**2, axis=1)
            cos2 = coord**2 / di[:, None]

            ctr = coord**2 / (n * eigval)

            return eigval, bs, cos2, ctr

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

class TestPCAStatisticsCalculator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame()

    def test_calculate_pca_statistics(self):
        pca_calculator = PCAStatisticsCalculator(self.df)
        result = pca_calculator.calculate_pca_statistics()

        self.assertIsNotNone(result)""")
            st.markdown("""-Création de l'instance de la classe :""")
            st.markdown(""" ```sql
                    pca_calculator = PCAStatisticsCalculator(Visualisation.df)
# Calcul des statistiques de l'ACP et récupération des résultats
eigval, bs, cos2, ctr = pca_calculator.calculate_pca_statistics()""")
            pca_calculator = PCAStatisticsCalculator(visualisation_instance.df)
            eigval, bs, cos2, ctr = pca_calculator.calculate_pca_statistics()
            st.markdown("""## **8-Visualisation des valeurs propres et des seuils**

Création d'un classe PCAGraphsVisualizer avec des méthodes qui produisent des graphiques pour visualiser les valeurs propres et les seuils, les COS² sur les deux premiers axes factoriels, ainsi que les CTR sur ces mêmes axes.

Ensuite nous avons créé notre **Unitest** pour la génération correcte des graphiques. Ils valident la représentation visuelle des données, assurant que les graphiques produits sont cohérents
""")
            st.markdown("""```sql
                    class PCAGraphsVisualizer:
    def __init__(self, eigval, bs, cos2, ctr):
        self.eigval = eigval
        self.bs = bs
        self.cos2 = cos2
        self.ctr = ctr

    def plot_eigenvalues_and_thresholds(self):
        try:
            p = len(self.eigval)
            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(1, p + 1), self.eigval, color='skyblue')
            plt.plot(np.arange(1, p + 1), self.bs, color='orange', marker='o', linestyle='-')
            plt.xlabel('Nombre de Facteurs')
            plt.ylabel('Valeurs Propres')
            plt.title('Valeurs propres et Seuils')
            plt.legend(['Seuils', 'Valeurs Propres'])
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_cos2(self):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.cos2[:, 0], self.cos2[:, 1], color='green')
            plt.xlabel('COS2 sur F1')
            plt.ylabel('COS2 sur F2')
            plt.title('COS2 sur les deux premiers axes factoriels')
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_ctr(self):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.ctr[:, 0], self.ctr[:, 1], color='red')
            plt.xlabel('CTR sur F1')
            plt.ylabel('CTR sur F2')
            plt.title('CTR sur les deux premiers axes factoriels')
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

class TestPCAGraphsVisualizer(unittest.TestCase):
    def setUp(self):
        self.eigval = np.array([1, 2, 3])  # Simuler des données pour les tests
        self.bs = np.array([0.5, 1.5, 2.5])  # Simuler des données pour les tests
        self.cos2 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Simuler des données pour les tests
        self.ctr = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])  # Simuler des données pour les tests

    def test_plot_eigenvalues_and_thresholds(self):
        graphs_visualizer = PCAGraphsVisualizer(self.eigval, self.bs, self.cos2, self.ctr)
        graphs_visualizer.plot_eigenvalues_and_thresholds()


""")
            st.markdown("""-Création de l'instance de la classe :""")
            st.markdown("""```sql 
                    # Création de l'instance de la classe PCAGraphsVisualizer
visualizer = PCAGraphsVisualizer(eigval, bs, cos2, ctr)""")
            visualizer = PCAGraphsVisualizer(eigval, bs, cos2, ctr)
            st.markdown("""-Ce graphique montre les valeurs propres et les seuils pour déterminer le nombre de facteurs significatifs""")
            fig=visualizer.plot_eigenvalues_and_thresholds()
            st.markdown("""```sql                    
# Affichage des graphiques
visualizer.plot_eigenvalues_and_thresholds()""")
            st.pyplot(fig)
            st.markdown(""" -Ce graphique représente la qualité de représentation des variables sur les composantes principales.""")
            st.markdown("""```sql
                    visualizer.plot_cos2()""")
            fig=visualizer.plot_cos2()
            st.pyplot(fig)
            st.markdown("""-Ce graphique représente la contribution des variables à la variance des composantes principales."""
        )
            st.markdown(""" ```sql 
                    visualizer.plot_ctr()""")
            fig=visualizer.plot_ctr()
            st.pyplot(fig)
        elif choice2=="Modelisation":
            st.markdown("""# **Prétraitement des Données**""")
            st.markdown("""```sql 
                        def clean_carburant(data):
    data['Carburant'].replace({'Essence M': 'Essence', 'ESSENCE': 'Essence','Essence B': 'Essence','HYBRIDE': 'Hybride','Hybride R': 'Hybride','Electrique': 'Hybride','Diesel M': 'Diesel','DIESEL': 'Diesel','Gpl':'Gaz'}, inplace=True)
    return data['Carburant'].unique()
def clean_type_moteur(data):
    data['Type_moteur'].replace({'460ch':'V12','8v':'V8','116ch':'V8','T':'TSI'},inplace=True)
    return data['Type_moteur'].unique()
def clean_transmission(data):
    data['Transmission'].replace({'AUTOMATIQUE':'Automatique','MANUELLE':'Manuelle'},inplace=True)
    return data['Transmission'].unique()""")
            st.markdown("""```sql
                        print(clean_carburant(data))""")
            st.write(clean_carburant(data))
            st.markdown("""```sql
            print(clean_type_moteur(data))""")
            st.write(clean_type_moteur(data))
            st.markdown("""```sql
            print(clean_transmission(data))""")
            st.write(clean_transmission(data))
            st.markdown("""```sql
                        class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Carburant': ['Essence M', 'Diesel M', 'HYBRIDE','hybride R','Essence','Diesel',,'Hybride','electrique','Gpl','Gaz'],
            'Type_moteur': ['460ch', '8v', 'T','V12','V8','TSI'],
            'Transmission': ['AUTOMATIQUE', 'MANUELLE', 'AUTOMATIQUE']
        })

    def test_clean_carburant(self):
        clean_carburant(self.data)
        expected_result = ['Essence', 'Diesel', 'Hybride','Gaz']
        self.assertListEqual(list(self.data['Carburant']), expected_result)

    def test_clean_type_moteur(self):
        clean_type_moteur(self.data)
        expected_result = ['V12', 'V8', 'TSI']
        self.assertListEqual(list(self.data['Type_moteur']), expected_result)

    def test_clean_transmission(self):
        clean_transmission(self.data)
        expected_result = ['Automatique', 'Manuelle']
        self.assertListEqual(list(self.data['Transmission']), expected_result)
""")

            
            st.markdown("""## **1- Classe de Traitement de Données pour la Modélisation de Prix**

La classe DataProcessor a pour but de prétraiter des données pour une tâche de prédiction de prix. Elle sépare les caractéristiques et la cible, divise les données en ensembles d'entraînement et de test, puis prétraite les caractéristiques numériques et catégorielles.

De plus, Nous avons ajouté des exceptions pour chaque méthodes pour garantir que les opérations sont effectuées de manière sûre
""")
            st.markdown("""```sql
                        class DataProcessor:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Les données doivent être de type DataFrame")
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def separate_features_target(self):
        if 'Prix' not in self.data.columns:
            raise ValueError("La colonne 'Prix' est introuvable dans les données")
        self.X = self.data.drop('Prix', axis=1)
        self.y = self.data['Prix']
        return self.X, self.y

    def split_train_test(self):
        if self.X is None or self.y is None:
            raise ValueError("Les caractéristiques et la cible doivent être séparées en premier")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=5
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Les données d'entraînement et de test doivent être divisées en premier")
        numeric_features = ['Chevaux', 'Cylindrée', 'CO2', 'KM', 'Année', 'Puissance', 'Puissance_fiscale']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_features = ['Marque', 'Modèle', 'Carburant', 'Type_moteur', 'Transmission']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        return self.X_train, self.X_test""")
            st.markdown("""## **2- Classe pour visualiser notre modèle de prédiction de prix**

Nous avons créé cette classe pour évaluer et visualiser les performances de notre modèle et nous avons ajouté les **test unitaires** pour que ca soit plus robuste et flexible. Elle fournit des fonctionnalités pour évaluer les modèles, afficher les coefficients et intercepts, ainsi que tracer **un graphique** des résidus.
""")
            st.markdown("""```sql
                        class ModelTrainerEvaluator:
    def __init__(self, model, X_test, y_test):
        if model is None or X_test is None or y_test is None:
            raise ValueError("Le modèle ou les données de test ne peuvent pas être vides")
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        return mse, r2

    def display_coefficients_and_intercept(self):
        if not hasattr(self.model, 'coef_') or not hasattr(self.model, 'intercept_'):
            raise AttributeError("Le modèle ne possède pas les attributs de coefficient et d'intercept")
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        return coefficients, intercept

    def plot_residuals(self):
        predictions = self.model.predict(self.X_test)
        residuals = self.y_test - predictions
        plt.scatter(predictions, residuals)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.show()
 """)
            st.markdown("""## **3-Instanciation et graph**""")
            st.markdown("""```sql
                        data = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/autosphere_data.csv', sep=';')
data_processor = DataProcessor(data)
X, y = data_processor.separate_features_target()
X_train, X_test, y_train, y_test = data_processor.split_train_test()
X_train_processed, X_test_processed = data_processor.preprocess_data()

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_processed, y_train)

evaluator = ModelTrainerEvaluator(linear_regression_model, X_test_processed, y_test)
evaluator.evaluate_model()
evaluator.display_coefficients_and_intercept()
evaluator.plot_residuals()""")
            linear_regression_model = LinearRegression()
            linear_regression_model.fit(X_train_processed, y_train)

            evaluator = ModelTrainerEvaluator(linear_regression_model, X_test_processed, y_test)
            evaluator.evaluate_model()
            evaluator.display_coefficients_and_intercept()
            st.markdown("""***Residual Plot***""")
            fig = evaluator.plot_residuals()
            st.pyplot(fig)

            st.markdown("""## **4- Test du modèle de prédiction de prix**""")
            st.markdown("""```sql
                        
# Charger de nouvelles données à partir d'un fichier CSV par exemple
new_data =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant', 'Chevaux','Type_moteur','Cylindrée','CO2','KM','Année','Transmission','Puissance','Puissance_fiscale'])
new_data.loc['0'] = ['RENAULT', 'KADJAR', 'Diesel', 115,'Blue dCi',1.5,138,13145,2022,'Automatique',84.64,6]
# Prétraiter les nouvelles données
new_data_processed = data_processor.preprocessor.transform(new_data)

# Faire des prédictions avec le modèle entraîné
predictions_new_data = linear_regression_model.predict(new_data_processed)

print(predictions_new_data)""")
            new_data =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant', 'Chevaux','Type_moteur','Cylindrée','CO2','KM','Année','Transmission','Puissance','Puissance_fiscale'])
            new_data.loc['0'] = ['RENAULT', 'KADJAR', 'Diesel', 115,'Blue dCi',1.5,138,13145,2022,'Automatique',84.64,6]
            new_data_processed = data_processor.preprocessor.transform(new_data)
            st.write(linear_regression_model.predict(new_data_processed))
            st.markdown("""## **5- Classe pour la prédiction des Chevaux**""")
            st.markdown("""```sql
                        class ChevauxProcc:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Les données doivent être de type DataFrame")
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def separate_features_target_ch(self):
        if 'Chevaux' not in self.data.columns:
            raise ValueError("La colonne 'Chevaux' est introuvable dans les données")
        self.X = self.data.drop('Chevaux', axis=1)
        self.y = self.data['Chevaux']
        return self.X, self.y

    def split_train_test_ch(self):
        if self.X is None or self.y is None:
            raise ValueError("Les caractéristiques et la cible doivent être séparées en premier")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=5
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data_ch(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Les données d'entraînement et de test doivent être divisées en premier")
        numeric_features = [ 'Cylindrée', 'Puissance_fiscale']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_features = ['Marque', 'Modèle', 'Carburant', 'Type_moteur', 'Transmission']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        return self.X_train, self.X_test""")
            st.markdown("""## **6-Instanciation et graph de la prediction de chevaux**""")
            st.markdown("""```sql
                        data = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/autosphere_data.csv', sep=';')
ch_processor = ChevauxProcc(data)
X_ch, y_ch = ch_processor.separate_features_target_ch()
X_train_ch, X_test_ch, y_train_ch, y_test_ch = ch_processor.split_train_test_ch()
X_train_processed_ch, X_test_processed_ch = ch_processor.preprocess_data_ch()

linear_regression_model_ch = LinearRegression()
linear_regression_model_ch.fit(X_train_processed_ch, y_train_ch)

evaluator_ch = ModelTrainerEvaluator(linear_regression_model_ch, X_test_processed_ch, y_test_ch)
evaluator_ch.evaluate_model()
evaluator_ch.display_coefficients_and_intercept()
evaluator_ch.plot_residuals()""")
            ch_processor = ChevauxProcc(data)
            X_ch, y_ch = ch_processor.separate_features_target_ch()
            X_train_ch, X_test_ch, y_train_ch, y_test_ch = ch_processor.split_train_test_ch()
            X_train_processed_ch, X_test_processed_ch = ch_processor.preprocess_data_ch()

            linear_regression_model_ch = LinearRegression()
            linear_regression_model_ch.fit(X_train_processed_ch, y_train_ch)

            evaluator_ch = ModelTrainerEvaluator(linear_regression_model_ch, X_test_processed_ch, y_test_ch)
            evaluator_ch.evaluate_model()
            evaluator_ch.display_coefficients_and_intercept()
            st.pyplot(evaluator_ch.plot_residuals())
            

            st.markdown("""## **7-Test de notre modèle de chevaux**""")
            st.markdown("""```sql
                        # Charger de nouvelles données à partir d'un fichier CSV par exemple
new_data_ch =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant','Type_moteur','Cylindrée','Année','Transmission','Puissance_fiscale'])
new_data_ch.loc['0'] = ['CITROEN', 'C3', 'Diesel','PureTech',1.2,2019,'MANUELLE',6]
# Prétraiter les nouvelles données
new_data_processed = ch_processor.preprocessor.transform(new_data)

# Faire des prédictions avec le modèle entraîné
predictions_new_data = linear_regression_model_ch.predict(new_data_processed)
print(predictions_new_data)""")
            
            new_data_ch =pd.DataFrame( index = ['0'], columns = ['Marque', 'Modèle', 'Carburant','Type_moteur','Cylindrée','Année','Transmission','Puissance_fiscale'])
            new_data_ch.loc['0'] = ['CITROEN', 'C3', 'Diesel','PureTech',1.2,2019,'MANUELLE',6]

            new_data_processed = ch_processor.preprocessor.transform(new_data)
            st.write(linear_regression_model_ch.predict(new_data_processed))
        else:
            st.markdown("""# Importation de la librairie StreamLit""")
            st.markdown("""```sql
                        import streamlit as st""")
            st.markdown("""# Mise en Place d'un Menu avec les deux estimateurs""")
            st.markdown("""```sql
                        def main():
    st.title("Estimateur de prix d'une voiture")
    menu = ["Estimer le prix de votre voiture", "Estimer le Prix d'une voiture sur Tayara","Visualiser le code"]""")
            st.markdown("""# Cas De l'Estimateur de votre voiture : """)
            st.markdown("""Nous Avons mis en place un formulaire dans lequel l'utilisateur
                        doit remplir ses information et où il a la possibilité de faire une approximation
                        de la Puissance DIN de sa voiture""")
            st.markdown("""```sql
                        if choice=="Estimer le prix de votre voiture":
        st.subheader("Estimer le prix de votre voiture")
        with st.form(key="esti_voit"):
            col1, col2 = st.columns([3, 1])
            with col1:
                Marque = st.text_input("Marque")
                Model=st.text_input("Modèle")
                Annee=int(st.number_input("Année",format='%d',min_value=0,max_value=2023))
                Carburant=st.selectbox("Carburant",Carb)
                Transmission=st.selectbox("Transmission",trans)
                Puissance_fiscale=int(st.number_input("Puissance Fiscale",format='%d',min_value=0))
                Type_moteur=st.selectbox("Type de moteur",moteur)
                Kilometrage=int(st.number_input("Kilométrage",format='%d',min_value=1000))
                Cylindree=st.number_input("Cylindrée (en L)",min_value=1.0,max_value=10.0)
                Chevaux=int(st.number_input("Puissance (en chevaux)",format='%d',min_value=0))
                Puissance=Chevaux*0.7355
                CO2=int((Puissance_fiscale-(Puissance/40)*1.6)*45)
            with col2:
                for i in range(49):
                    st.write("")
                button_clicked = st.form_submit_button("Approximer")
            submit_button = st.form_submit_button(label='Estimer')""")
            st.markdown("""L'approximation de la puissance DIN""")
            st.markdown("""```sql
                        if button_clicked:
                st.success("Votre voiture fait environ : {} ch".format(estimer_ch(Marque, Model, Carburant,Type_moteur,Cylindree,Annee,Transmission,Puissance_fiscale)))""")
            st.markdown("""L'approximation du prix""")
            st.markdown("""```sql
                        if submit_button:
            st.success("Votre voiture vaut environ : {} dt".format(estimer_prx(Marque, Model, Carburant, Chevaux,Type_moteur,Cylindree,CO2,Kilometrage,Annee,Transmission,Puissance,Puissance_fiscale)))""")
            st.markdown("""# Cas De Tayara : """)
            st.markdown("""Dans Le cas de l'estimateur Tayara, Nous verifions tout d'abord que nous sommes
                        bien sur le site Tayara.tn dans la catégorie Voitures, et nous procédons de la même manière 
                        que dans le cas précédent""")
            st.markdown("""```sql
                        st.subheader("Estimer le Prix d'une voiture sur Tayara")
        with st.form(key="esti_tay"):
            url = st.text_input("URL Tayara.tn")
            submit_button2 = st.form_submit_button(label='Estimer')
        if submit_button2:
            if ("tayara.tn" in url and "Voitures" in url):
                spec=[]
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                l_specs=soup.find_all('span',class_="text-gray-700/80 text-xs md:text-sm lg:text-sm font-semibold")
                for i in range(len(l_specs)):
                    spec.append(l_specs[i].text)
                kilometrage=spec[0]
                Transmission=spec[3]
                Annee=int(spec[4])
                Cylindree=float(re.sub(r"[a-zA-Z]", "", spec[5]))
                Marque=spec[6]
                Modele=spec[7]
                Chevaux_fisc=int(spec[8])
                Carburant=spec[10]
                Type=spec[9]
                Chevaux=estimer_ch(Marque,Modele,Carburant,Type,Cylindree,Annee,Transmission,Chevaux_fisc)
                Puissance=Chevaux*0.7355
                CO2=int((Chevaux_fisc-(Puissance/40)*1.6)*45)
                st.success("Votre voiture vaut environ : {} dt".format(estimer_prx(Marque,Modele,Carburant,Chevaux,Type,Cylindree,CO2,kilometrage,Annee,Transmission,Puissance,Chevaux_fisc)))
            else :
                st.error("Vous devez saisir un lien du site tayara.tn de la catégorie Voitures ")""")

if __name__ =="__main__":
    main()
