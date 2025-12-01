import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st

def obtner_datos(stocks):
    df = yf.download(stocks, start = '2010-01-01', end= '2025-01-01'  )['Close']
    return df 

tickers = {
    "regiones": ["SPLG", "EWC", "IEUR", "EEM", "EWJ"],
    "sectores": ["XLC", "XLY", "XLP", "XLE", "XLF",
                 "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
}

df = obtner_datos(tickers['regiones'])


def rendimientos(df):
    rendimientos = df.pct_change().dropna()
    rendimientos_esperados = performance(rendimientos)*252
    covarianzas = rendimientos.cov()*252 
    return rendimientos, rendimientos_esperados, covarianzas

# Función promedio
def performance(performance_Stocks):
    return (performance_Stocks.sum())/(performance_Stocks.count())

def portafolio_rendimiento(pesos, rendimientos_esperados):
    return np.sum(pesos * rendimientos_esperados)

def portafolio_volatilidad(pesos, covarianza):
    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))


def simular_portafolios(rendimientos_esperados, covarianzas, num_portafolios =2000, rrf = 0.404):
    num_activos = len(rendimientos_esperados)
    pesos = np.zeros((num_portafolios, num_activos))
    expected_Returns = np.zeros(num_portafolios)
    expected_Volatility = np.zeros(num_portafolios)
    sharpe_ratio = np.zeros(num_portafolios)
    
    for k in range(num_portafolios):
        #Generar un vector de pesos
        w = np.random.random(num_activos)
        w = w / np.sum(w)
        pesos[k,:] = w
        #Generar tabla de rendimientos
        expected_Returns[k]= portafolio_rendimiento(w, rendimientos_esperados)
        #Generar tabla de volatilidades
        expected_Volatility[k] = portafolio_volatilidad(w, covarianzas)
        #Calculo SHarpe Ratio
        sharpe_ratio[k] = (expected_Returns[k] - rrf)/ expected_Volatility[k]
    return pesos, expected_Returns, expected_Volatility, sharpe_ratio



def frontera_eficiente(rendimientos_esperados, covarianza, p= 100):
    num_activos = len(rendimientos_esperados)

    # Función objetivo: minimizar la volatilidad
    def portafolio_vol(pesos):
        return portafolio_volatilidad(pesos, covarianza)
    
    rendimientos_objetivo = np.linspace(min(rendimientos_esperados), max(rendimientos_esperados), p)
    frontera_volatilidad = []
    frontera_pesos = []
    frontera_retornos = []
    
    for r in rendimientos_objetivo:
        # Restricciones: suma de pesos = 1, rendimiento objetivo
        restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                     {'type': 'eq', 'fun': lambda x: portafolio_rendimiento(x, rendimientos_esperados) - r})
        # Límites de los pesos: entre 0 y 1
        limites = tuple((0, 1) for _ in range(num_activos))
        # Optimización
        resultado = minimize(portafolio_vol, num_activos * [1. / num_activos,], method='SLSQP', bounds=limites, constraints=restricciones)
        
        if resultado.success:
            frontera_volatilidad.append(portafolio_volatilidad(resultado.x, covarianza))
            frontera_pesos.append(resultado.x)    
            frontera_retornos.append(portafolio_rendimiento(resultado.x, rendimientos_esperados))
    return frontera_volatilidad, frontera_retornos, frontera_pesos
        

def portafolio_min_var(rendimientos_esperados, covarianza):
    num_activos = len(rendimientos_esperados)

    def objetivo_var(pesos):
        return portafolio_volatilidad(pesos, covarianza)
    
    # Restricciones: suma de pesos = 1, rendimiento objetivo
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Límites de los pesos: entre 0 y 1
    limites = tuple((0, 1) for _ in range(num_activos))
        # Optimización
    resultado = minimize(objetivo_var, num_activos * [1. / num_activos,], method='SLSQP', bounds=limites, constraints=restricciones)

    return resultado
        
def sharpe_ratio(pesos, rendimientos_esperados, covarianza, risk_free_rate=0.044):
    return (portafolio_rendimiento(pesos, rendimientos_esperados) - risk_free_rate) / portafolio_volatilidad(pesos, covarianza)

def portafolio_tangente(rendimientos_esperados, covarianza, rf):
    num_activos = len(rendimientos_esperados)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_activos))

    resultado = minimize(lambda x: -sharpe_ratio(x, rendimientos_esperados, covarianza, rf), num_activos * [1. / num_activos,], method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado

def graficar(
    vol_mc, ret_mc, sharpe_mc,                     
    vol_fe, ret_fe,                               
    resultado_pmv, resultado_sharpe,               
    rendimientos_esperados, covarianza, tickers    
):
 
    pesos_pmv = resultado_pmv.x
    pesos_sharpe = resultado_sharpe.x

    vol_pmv = np.sqrt(np.dot(pesos_pmv.T, np.dot(covarianza, pesos_pmv)))
    ret_pmv = np.sum(pesos_pmv * rendimientos_esperados)

    vol_sharpe = np.sqrt(np.dot(pesos_sharpe.T, np.dot(covarianza, pesos_sharpe)))
    ret_sharpe = np.sum(pesos_sharpe * rendimientos_esperados)

  
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=vol_fe,
        y=ret_fe,
        mode="lines",
        line=dict(width=3, color="blue"),
        name="Frontera Eficiente"
    ))

    fig.add_trace(go.Scatter(
        x=[vol_pmv],
        y=[ret_pmv],
        mode="markers",
        marker=dict(size=14, color="red", symbol="star"),
        name="Portafolio Min Var",
        text=[f"<b>{tickers[i]}</b>: {pesos_pmv[i]:.2%}" for i in range(len(tickers))],
        hovertemplate="Vol = %{x:.2%}<br>Ret = %{y:.2%}<br>%{text}<extra></extra>"
    ))

    
    fig.add_trace(go.Scatter(
        x=[vol_sharpe],
        y=[ret_sharpe],
        mode="markers",
        marker=dict(size=14, color="black", symbol="diamond"),
        name="Portafolio Máx Sharpe",
        text=[f"<b>{tickers[i]}</b>: {pesos_sharpe[i]:.2%}" for i in range(len(tickers))],
        hovertemplate="Vol = %{x:.2%}<br>Ret = %{y:.2%}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title="Portafolios Óptimos: Markowitz, Mínima Varianza y Máximo Sharpe",
        xaxis_title="Volatilidad",
        yaxis_title="Rendimiento Esperado",
        template="plotly_white",
        width=950,
        height=650,
        legend=dict(x=0.75, y=0.05)
    )