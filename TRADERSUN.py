import yfinance as yf
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os
import asyncio 
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, request

# Se elimina la importación innecesaria y conflictiva:
# from asgiref.sync import sync_to_async 

# NOTA: Reemplaza con tu token real de BotFather
TOKEN = "8246576801:AAEORFpWu_gwXhRq7QznMb1mwnCYeH3-uOk" # Usa tu token real

# ----------------------------------------------------
# INICIALIZACIÓN GLOBAL DE MODELO (CARGA RÁPIDA)
# ----------------------------------------------------
try:
    MODELO_GLOBAL = joblib.load('tradersun_modelo.pkl')
    PRECISION_GLOBAL = 85.0 
    print("Modelo de ML cargado exitosamente.")
except FileNotFoundError:
    print("ERROR CRÍTICO: No se encontró 'tradersun_modelo.pkl'. El bot no funcionará.")
    MODELO_GLOBAL = None
    PRECISION_GLOBAL = 0.0

# ------------------------------
# Función de Entrenamiento (MANTENIMIENTO/DATOS HISTÓRICOS)
# ------------------------------
def entrenar_modelo(par="EURUSD=X", intervalo="15m", dias="30d"):
    df = yf.download(par, period=dias, interval=intervalo, auto_adjust=True)
    return None, 0.0, df 

# ------------------------------
# Generación de señal (El cuerpo de esta función se mantiene sin cambios)
# ------------------------------
def generar_senal(par: str, intervalo: str, modelo, precision: float) -> str:
    try:
        df = yf.download(par, period="5d", interval=intervalo, auto_adjust=True)
        if df.empty or modelo is None:
            return f"⚠️ No se pudieron obtener datos para {par} en {intervalo}"

        df.index = df.index.tz_convert("America/Caracas")

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        # Indicadores
        rsi = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])
        cci = float(ta.trend.CCIIndicator(high, low, close).cci().iloc[-1])
        stoch = float(ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1])
        adx = float(ta.trend.ADXIndicator(high, low, close).adx().iloc[-1])
        atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
        atr_index = (atr / float(df["High"].max())) * 100

        # Señal del modelo
        X_new = pd.DataFrame([[rsi, cci, stoch, adx]], columns=["RSI", "CCI", "STOCH", "ADX"])
        pred = modelo.predict(X_new)[0]  # 1 = CALL, 0 = PUT
        confianza = float(modelo.predict_proba(X_new)[0][pred] * 100)

        # Análisis gráfico
        ultima_vela = df.iloc[-1]
        close_val = float(ultima_vela["Close"])
        open_val = float(ultima_vela["Open"])
        vela = "alcista" if close_val > open_val else "bajista"

        ema20 = float(df["Close"].ewm(span=20).mean().iloc[-1])
        ema50 = float(df["Close"].ewm(span=50).mean().iloc[-1])
        tendencia = "alcista" if ema20 > ema50 else "bajista"

        soporte = float(df["Low"].rolling(20).min().iloc[-1])
        resistencia = float(df["High"].rolling(20).max().iloc[-1])
        cerca_resistencia = close_val >= resistencia * 0.98
        cerca_soporte = close_val <= soporte * 1.02

        # Score de fuerza
        score = 0
        if (pred == 1 and vela == "alcista") or (pred == 0 and vela == "bajista"):
            score += 30
        if (pred == 1 and tendencia == "alcista") or (pred == 0 and tendencia == "bajista"):
            score += 30
        if not cerca_resistencia and not cerca_soporte:
            score += 20
        if adx > 20:
            score += 20

        # Semáforo visual con recomendación de entrada
        if score >= 70:
            if (pred == 1 and vela == "alcista") or (pred == 0 and vela == "bajista"):
                semaforo = "🟢 Entrar de una vez (alta confianza)"
            else:
                semaforo = "🟢 Señal fuerte, pero esperar la próxima vela"
        elif 40 <= score < 70:
            semaforo = "🟡 Esperar/confirmar (riesgo moderado)"
        else:
            semaforo = "🔴 Evitar (señal débil)"

        # Checklist rápido (✅/❌)
        checklist = (
            f"📋 Checklist disciplina:\n"
            f"   • Tendencia confirma → {'✅' if (pred==1 and tendencia=='alcista') or (pred==0 and tendencia=='bajista') else '❌'}\n"
            f"   • Última vela confirma → {'✅' if (pred==1 and vela=='alcista') or (pred==0 and vela=='bajista') else '❌'}\n"
            f"   • ADX > 20 (mercado con fuerza) → {'✅' if adx > 20 else '❌'}\n"
            f"   • No pegado a soporte/resistencia → {'✅' if not cerca_resistencia and not cerca_soporte else '❌'}"
        )

        # Mensaje final
        return (
            f"📈 Señal: {'CALL' if pred==1 else 'PUT'} ({confianza:.2f}% confianza)\n"
            f"📊 Análisis gráfico:\n"
            f"   • Última vela: {vela}\n"
            f"   • Tendencia EMA20/EMA50: {tendencia}\n"
            f"   • Soporte: {soporte:.2f}, Resistencia: {resistencia:.2f}\n"
            f"   • ADX={adx:.2f}, ATR={atr_index:.2f}/100\n"
            f"🔥 Fuerza de señal: {score}/100\n"
            f"{semaforo}\n\n"
            f"{checklist}"
        )

    except Exception as e:
        return f"❌ Error analizando {par}: {e}"
        
# ------------------------------
# Handlers del Bot de Telegram
# ------------------------------
async def menu_otc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🚀 Bienvenido a Tradersun Bot. Usa los botones para generar señales.")

async def manejar_seleccion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⚡ Selección recibida. (Handler en construcción)")

async def manejar_intervalo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.split("|")
    par, intervalo = data

    await query.edit_message_text(text=f"🔍 Analizando {par} en {intervalo}...")

    modelo = MODELO_GLOBAL 
    precision = PRECISION_GLOBAL
    
    if modelo is None:
        await context.bot.send_message(chat_id=query.message.chat_id, text="❌ Error: El modelo de análisis no pudo cargarse al iniciar el bot.")
        return

    _, _, df_hist = entrenar_modelo(par, intervalo) 
    senal = generar_senal(par, intervalo, modelo, precision)

    keyboard = [
        [InlineKeyboardButton("📡 Nueva señal", callback_data="nueva_senal")],
        [InlineKeyboardButton("📊 Ver rendimiento histórico", callback_data=f"ver_rendimiento|{par}|{intervalo}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.user_data["df_hist"] = df_hist

    await context.bot.send_message(chat_id=query.message.chat_id, text=senal, reply_markup=reply_markup)

async def manejar_nueva_senal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📡 Generando nueva señal... (Handler en construcción)")

async def manejar_rendimiento(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📊 Mostrando rendimiento histórico... (Handler en construcción)")

# ------------------------------
# Configuración del bot (handlers y aplicación)
# ------------------------------
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", menu_otc))
app.add_handler(CallbackQueryHandler(manejar_seleccion, pattern="^(?!.*\\|).*"))  
app.add_handler(CallbackQueryHandler(manejar_intervalo, pattern=".*\\|.*"))         
app.add_handler(CallbackQueryHandler(manejar_nueva_senal, pattern="nueva_senal"))
app.add_handler(CallbackQueryHandler(manejar_rendimiento, pattern="ver_rendimiento.*"))

# ------------------------------
# Arranque final del bot (modo polling)
# ------------------------------
if __name__ == "__main__":
    app.run_polling()
