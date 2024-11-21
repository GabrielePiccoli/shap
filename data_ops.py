from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import numpy as np
import math
from scipy.special import roots_chebyt
import pandas as pd
from pyDOE import lhs
from scipy.optimize import Bounds
from scipy.stats.distributions import triang as dist

def website_call(dataset):
    df=pd.DataFrame()
    constant=0
    mse=[]
    driver = webdriver.Chrome()
    # dataset è una lista di data, ciascuno dei quali rappresenta una combinazione di parametri
    # ogni data è una lista di coppie (nome_parametro, valore_assegnato)
    for data in dataset:
        #query verrà usata nella chiamata al sito
        query = ""
        #entry sarà l'effettiva riga del dataframe
        entry = []
        # prendo una coppia alla volta e costruisco query e entry
        for (param_name, param_value) in data:
            entry.append(param_value)
            if param_value != '':
                query += f'&{param_name}={param_value}'
        
        #appendo la query all'url
        url = f'https://en-roads.climateinteractive.org/scenario.html?v=24.6.0'+query;
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        print(url)

        try:
            close_element = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'close'))) #chiude la schermata che si apre non appena carica la pagina
            close_element.click()
            menu_element = wait.until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "div.actions-button")))
            menu_element[1].click() #ci sono due menu definiti dalla stessa classe, prendo il secondo
            span_element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//span[text()='Copy Data to Clipboard']")))
            span_element.click()

            df = pd.read_clipboard(skiprows=1,sep='\t',usecols=['Current Scenario'])
            values = df['Current Scenario'].iloc[25:].values #consideriamo dal 2025 in poi
            # aggiungo l'uscita alla entry
            error=values-constant
            entry[-1] = error
            #per la costruzione del dataframe è comodo che mse sia una lista di tuple in cui ciascuna tupla sarà intesa come elemento del dataframe
            tuple_entry = tuple(entry)
            mse.append(tuple_entry)


        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    driver.quit()
    return mse

def format_data(df):
    y = []
    for i in range(len(df)):
        data = df.iloc[i, -1]
        data = data[1:-1]
        data = data.replace("\n", "")
        data = data.replace(",", "")
        data = list(data.split(" "))
        ret_data = []
        for i in range(0,len(data)):
            try:
                if not (data[i].isspace() or data[i] == ""):
                    ret_data.append(float(data[i]))
            except:
                print("Not a number: " + data[i])
        y.append(ret_data)
    return y

def normalize(df, intervals, years, year_cols_indexes):
    for c in df.columns[year_cols_indexes]:
        lower_bound = years[0]
        upper_bound = years[-1]
        df[c] = (df[c] - lower_bound) / (upper_bound - lower_bound)
    
    for value in intervals.values():
        c = value[0]
        upper_bound = value[1][1]
        lower_bound = value[1][3]
        df[c] = (df[c] - lower_bound) / (upper_bound - lower_bound)
        
    return df

def denormalize(df, intervals, years, year_cols_indexes):
    year_cols = df.columns[year_cols_indexes]
    print(year_cols)
    for c in df.columns[year_cols_indexes]:
        lower_bound = years[0]
        upper_bound = years[-1]
        
        df[c] = df[c] * (upper_bound - lower_bound) + lower_bound
    
    for value in intervals.values():
        c = value[0]
        upper_bound = value[1][1]
        lower_bound = value[1][3]
        df[c] = df[c] * (upper_bound - lower_bound) + lower_bound
        
    return df

def normalize_y(y, norm_fact):
    y = np.asarray(y)/norm_fact
    y = list(y)
    return y

def denormalize_y(y, norm_fact):
    y = np.asarray(y) * norm_fact
    y = list(y)
    return y

def prepare_for_model(df, y, current_year, year_dep_columns, n_points):
    # Classic preparation
    df = df.fillna(0)
    df.drop(columns=['funct'], inplace=True)

    years = list(np.linspace(start=0, stop=1, endpoint=False, num=len(y[0])))
    years = sample(years,n_points)

    print(years)

    for i in range(len(y)):
        y[i] = sample(y[i], n_points)

    dataset = []
    for i in range(len(df)):
        features = list(np.asarray(df.iloc[i].values))
        prep_features = prepare_input(features, year_dep_columns, years)
        dataset.append(prep_features)
    
    dataset = np.asarray(dataset)
    y = np.asarray(y)

    return dataset, y

def prepare_input_with_years(input, year_dep_columns, years): 
    
    ret_input_list = []
    for year in years:
        entry = []
        for i in range(len(input)):
            if i in year_dep_columns:
                if (input[i+1] > year):
                    entry.append(0)
                else:
                    entry.append(input[i])
            else:
                entry.append(input[i])

        ret_input_list.append(entry)
    return ret_input_list

def prepare_input(input, year_dep_columns, years): 
    ret_input_list = []
    for year in years:
        if np.max(years) > 1:
            year = year / np.max(years)
        entry = []
        for i in range(len(input)):
            if i in year_dep_columns:
                if (input[i+1] > year):
                    entry.append(0)
                else:
                    entry.append(input[i])
            elif not (i-1) in year_dep_columns:
                entry.append(input[i])
        ret_input_list.append(entry)
    return ret_input_list

def find_batch_size(length_ds, lower_bound = 30, upper_bound = 50):
    divs = []
    for i in range(1, int(math.sqrt(length_ds)) + 1):
        if length_ds % i == 0:
            divs.append(i)
            d = i
            if d > lower_bound and d < upper_bound:
                return d
            if i != length_ds // i:
                d = length_ds // i
                if d > lower_bound and d < upper_bound:
                    return d
    return 1

def sample(signal, n_points):
    signal = list(signal)

    if n_points == len(signal):
        return signal

    nodes,weights = roots_chebyt(n_points)
    a = 0
    b = len(signal)
    nodes = (a + b)/2 + (b-a)/2*nodes

    points = [signal[int(n)] for n in nodes]
    
    return points

# Ritorna un dizionario in cui alla descrizione di ogni variabile sono legati:
# - il nome del parametro che ne modifica il valore
# - un vettore contenente le soglie applicabili 
# - il parametro dell'anno di applicazione

def create_intervals(file_path):
    intervals = {}
    tests = pd.read_excel(file_path)
    
    for index in tests.index:
            param_name = tests.loc[index,'Descrizione']
            param_level = tests.loc[index,'Parametro']
            level_high = tests.loc[index,'High']
            level_med = tests.loc[index,'Med']
            level_low = tests.loc[index,'Low']
            param_year = tests.iloc[index,5]
            intervals[param_name] = (param_level, ['', level_high, level_med, level_low], param_year)                                         
    return intervals

# Dato un number da 0 a 1 e un array, ritorna un elemento dell'array in posizione number*len(array)
def choose(number, array):
    if number >= 1:
        number = number % 1
        return choose(number, array)
    if number < 0: 
        return choose(-number, array)
    
    index = int(number*len(array))
    return array[index]

# Dato il dizionario costruito con create_intervals effettua un campionamento con la tecnica latin hypercube ritornando n_elems campioni
# IL campionamento è fatto sull'intervallo continuo [low, high]
def create_dataset(intervals, n_elems, years, continuous = True):
    dataset = []
    n_vars = 31
    lhd = lhs(n_vars, samples=n_elems, criterion='maximin')
    year_dep_columns = [2*x for x in range(0,12)] + [26, 28]
    year_col_indexes  = np.asarray(year_dep_columns) + 1
    elems = np.zeros(shape=lhd.shape)
    for i in range(len(elems[0])):
        if i in year_col_indexes:
            elems[:,i] = list(1-dist(c= 1).ppf(lhd[:,i]))
        else:
            elems[:,i] = list(dist(c= 1).ppf(lhd[:,i]))
 
    # Prendo uno alla volta tutti i campioni
    for el in elems:
        k = 0
        entry = []
        # Prendo uno alla volta tutte le variabili di decisione
        for value in intervals.values():
            # Il primo elemento di value è il nome del parametro
            param_level = value[0]
            # Sceglie un valore continuo tra le soglie low e high usando il k-esimo valore estratto da latin hypercube
            if continuous:
                array = ['']+list(np.linspace(value[1][-1], value[1][1]))
            else:
                array = [value[1][0],value[1][3],value[1][2],value[1][1]]
            level_value = choose(el[k], array)
            k+=1
            entry.append((param_level, level_value))
            # Il terzo elemento di value è il nome del parametro dell'anno di applicazione
            param_year = value[2]
            # Il parametro non è definito per tutte le variabili per cui verifica che esista
            if param_year == param_year:
                # Sceglie un valore tra gli anni possibili usando il k-esimo valore estratto da latin hypercube
                year_value = choose(el[k], years)
                k+=1
                entry.append((param_year,year_value))

        # Aggiunge la colonna per la funzione che verrà calcolata facendo la query a enroad
        entry.append(('funct', ''))
        dataset.append(entry)
    return dataset

# Prende dal file excel i nomi dei parametri
def get_columns(file_path):
    columns = []
    tests = pd.read_excel(file_path)      
    for index in tests.index:
        # nome del parametro che indica il livello
        param_level=tests.loc[index,'Parametro']
        columns.append(param_level)
        # nome del parametro che indica l'anno
        param_year=tests.iloc[index, 5]
        # se param_year è Nan (non è definito nel file excel), la condizione non è soddisfatta e non lo aggiungo alle colonne
        if param_year == param_year:
            columns.append(param_year)
    # preparo la colonna che ospiterà l'andamento di CO2
    columns.append('funct')
    return columns

def get_bounds(file_path, levels = None, years = None):
    lb = []
    ub = []
    tests = pd.read_excel(file_path)
    if years == None:
        years = [0,1]
    for index in tests.index:
        if levels == None:
            lb_i = 0
            ub_i = 1
        else:
            lb_i = tests.loc[index, levels[0]]
            ub_i = tests.loc[index, levels[-1]]
        
        if lb_i > ub_i: lb_i, ub_i = ub_i, lb_i
        lb.append(lb_i)
        ub.append(ub_i)
        param_year=tests.iloc[index, 5]
        if param_year == param_year:
            lb.append(years[0])
            ub.append(years[-1])
    bounds = Bounds(lb,ub)
    return bounds

# Dato il dizionario costruito con create_intervals effettua un campionamento con la tecnica latin hypercube ritornando n_elems campioni
# I livelli campionati sono solo quelli indicati dalle soglie
def create_discrete_dataset(intervals, n_elems, years):
    dataset = []
    n_vars = 31
    lhd = lhs(n_vars, samples=n_elems, criterion='maximin')
    year_dep_columns = [2*x for x in range(0,12)] + [26, 28]
    year_col_indexes  = np.asarray(year_dep_columns) + 1
    elems = np.zeros(shape=lhd.shape)
    for i in range(len(elems[0])):
        if i in year_col_indexes:
            elems[:,i] = list(1-dist(c= 1).ppf(lhd[:,i]))
        else:
            elems[:,i] = list(dist(c= 1).ppf(lhd[:,i]))
 
 
    # Prendo uno alla volta tutti i campioni
    for el in elems:
        k = 0
        entry = []
        # Prendo uno alla volta tutte le variabili di decisione
        for value in intervals.values():
            # Il primo elemento di value è il nome del parametro
            param_level = value[0]
            # Sceglie un valore tra le soglie usando il k-esimo valore estratto da latin hypercube
            array = [value[1][0],value[1][3],value[1][2],value[1][1]]
            level_value = choose(el[k], array)
            k+=1
            entry.append(level_value)
            # Il terzo elemento di value è il nome del parametro dell'anno di applicazione
            param_year = value[2]
            # Il parametro non è definito per tutte le variabili per cui verifica che esista
            if param_year == param_year:
                # Sceglie un valore tra le soglie usando il k-esimo valore estratto da latin hypercube
                year_value = choose(el[k], years)
                k+=1
                entry.append(year_value)

        # Aggiunge la colonna per la funzione che verrà calcolata facendo la query a enroad
        entry.append('')
        dataset.append(entry)
    return dataset


def create_dataset_direct(intervals, n_elems, years, continuous = True, distributed = False):
    dataset = []
    n_vars = 31
    lhd = lhs(n_vars, samples=n_elems, criterion='maximin')
    year_dep_columns = [2*x for x in range(0,12)] + [26, 28]
    year_col_indexes  = np.asarray(year_dep_columns) + 1
    if distributed:
        elems = np.zeros(shape=lhd.shape)
        for i in range(len(elems[0])):
            if i in year_col_indexes:
                elems[:,i] = list(1-dist(c = 1).ppf(lhd[:,i]))
            else:
                elems[:,i] = list(dist(c = 1).ppf(lhd[:,i]))
    else:
        elems = np.copy(lhd)
 
    # Prendo uno alla volta tutti i campioni
    for el in elems:
        k = 0
        entry = []
        # Prendo uno alla volta tutte le variabili di decisione
        for value in intervals.values():
            # Il primo elemento di value è il nome del parametro
            param_level = value[0]
            # Sceglie un valore continuo tra le soglie low e high usando il k-esimo valore estratto da latin hypercube
            if continuous:
                array = ['']+list(np.linspace(value[1][-1], value[1][1]))
                level_value = el[k] * (value[1][1] - value[1][-1]) + value[1][-1]
            else:
                array = [value[1][0],value[1][3],value[1][2],value[1][1]]
                level_value = choose(el[k], array)
            k+=1
            entry.append((param_level, level_value))
            # Il terzo elemento di value è il nome del parametro dell'anno di applicazione
            param_year = value[2]
            # Il parametro non è definito per tutte le variabili per cui verifica che esista
            if param_year == param_year:
                # Sceglie un valore tra gli anni possibili usando il k-esimo valore estratto da latin hypercube
                year_value = el[k] * (years[-1] - years[0]) + years[0]
                k+=1
                entry.append((param_year,year_value))

        # Aggiunge la colonna per la funzione che verrà calcolata facendo la query a enroad
        entry.append(('funct', ''))
        dataset.append(entry)
    return dataset
""" def format_data(df):
    y = []
    for i in range(len(df)):
        data = df.iloc[i, -1]
        data = data[1:-1]
        data = data.replace("\n", "")
        data = data.replace(",", "")
        data = list(data.split(" "))
        ret_data = []
        for i in range(0,len(data)):
            try:
                if not (data[i].isspace() or data[i] == ""):
                    ret_data.append(float(data[i]))
            except:
                print("Not a number: " + data[i])
        y.append(ret_data)
    return y

def prepare_for_model(df,y):
    # Classic preparation
    df = df.fillna(0)
    df.drop(columns=['funct'], inplace=True)

    cols_numeriche = df.select_dtypes(include=[np.number]).columns
    df[cols_numeriche] = df[cols_numeriche].applymap(lambda x: x - current_year if x > 2000 else x)

    x = list(np.linspace(0, len(y[0])-1, len(y[0]), dtype=np.int64))
    x = sample(x,step)
    
    for i in range(len(y)):
        y[i] = sample(y[i], step)

    dataset = []
    for i in range(len(df)):
        features = np.asarray(df.iloc[i].values)
        features_list = []
        for year in x:
            feature_year = np.copy(features)
            for i in range(1,n_features_time_var,2):
                if feature_year[i] > year:
                    feature_year[i-1] = 0
            features_list.append(np.copy(feature_year))
        dataset.append(features_list)
    dataset = np.asarray(dataset)
    y = np.asarray(y)
    return dataset, y


def find_batch_size(length_ds, lower_bound = 30, upper_bound = 50):
    divs = []
    for i in range(1, int(math.sqrt(length_ds)) + 1):
        if length_ds % i == 0:
            divs.append(i)
            d = i
            if d > lower_bound and d< upper_bound:
                return d
            if i != length_ds // i:
                d = length_ds // i
                if d > lower_bound and d< upper_bound:
                    return d
    return 1

def sample(signal, n_points):
    signal = list(signal)
    nodes,weights = roots_chebyt(n_points)
    a = 0
    b = len(signal)
    nodes = (a + b)/2 + (b-a)/2*nodes

    points = [signal[int(n)] for n in nodes]
    
    return points
"""