import yfinance as yf
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

<<<<<<< HEAD
import os from dotenv import load_dotenv # Cargar variables desde .env load_dotenv() # Leer el token desde la variable de entorno TOKEN = os.getenv("TOKEN")  # Reemplaza con tu token real de BotFather
=======
TOKEN = ""  # Reemplaza con tu token real de BotFather
>>>>>>> a7e7edd3721ecb31a71f39d6e9f07d77157bea6f

# ------------------------------
# Entrenamiento del modelo
# ------------------------------
def entrenar_modelo(par="EURUSD=X", intervalo="15m", dias="30d"):
    df = yf.download(par, period=dias, interval=intervalo, auto_adjust=True)
    if df.empty:
        return None, 0.0, None

    df.index = df.index.tz_convert("America/Caracas")

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # Indicadores técnicos
    df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    df["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
    df["STOCH"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["ADX"] = ta.trend.ADXIndicator(high, low, close).adx()

    # 🔎 Calcular ATR y normalizar a índice 0–100
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["ATR_Index"] = (atr / atr.max()) * 100

    df = df.dropna()
    df["target"] = np.where(df["Close"].values > df["Open"].values, 1, 0)

    X = pd.DataFrame(df[["RSI", "CCI", "STOCH", "ADX"]].values,
                     columns=["RSI", "CCI", "STOCH", "ADX"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestClassifier(n_estimators=120, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred) * 100

    return modelo, precision, df
# ------------------------------
# Generación de señal con análisis gráfico automático + semáforo + checklist
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
# Rendimiento histórico (gráfico)
# ------------------------------
def generar_grafico_rendimiento(df: pd.DataFrame, par: str, intervalo: str) -> BytesIO:
    df = df.copy()
    df["target"] = np.where(df["Close"].values > df["Open"].values, 1, 0)
    df["pred_dummy"] = np.where(df["RSI"] > 50, 1, 0)
    df["acierto"] = (df["target"] == df["pred_dummy"]).astype(int)
    df["rolling_acc"] = df["acierto"].rolling(50).mean() * 100

    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["rolling_acc"], label="Precisión rolling (RSI>50 ref.)", color="#2b8a3e")
    plt.axhline(50, color="#999", linestyle="--", linewidth=1)
    plt.axhline(70, color="red", linestyle="--", linewidth=1, label="RSI 70 (sobrecompra)")
    plt.axhline(30, color="blue", linestyle="--", linewidth=1, label="RSI 30 (sobreventa)")
    plt.title(f"Rendimiento histórico - {par} ({intervalo})")
    plt.ylabel("Precisión (%)")
    plt.xlabel("Tiempo")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf
# ------------------------------
# Menú de pares con banderas
# ------------------------------
async def menu_otc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🇺🇸/🇯🇵 USD/JPY OTC", callback_data="USDJPY=X"),
         InlineKeyboardButton("🇬🇧/🇺🇸 GBP/USD OTC", callback_data="GBPUSD=X")],
        [InlineKeyboardButton("🇪🇺/🇺🇸 EUR/USD OTC", callback_data="EURUSD=X"),
         InlineKeyboardButton("🇦🇺/🇨🇭 AUD/CHF OTC", callback_data="AUDCHF=X")],
        [InlineKeyboardButton("🇺🇸/🇨🇦 USD/CAD OTC", callback_data="USDCAD=X"),
         InlineKeyboardButton("🇬🇧/🇨🇦 GBP/CAD OTC", callback_data="GBPCAD=X")],
        [InlineKeyboardButton("🇦🇺/🇪🇺 EUR/AUD OTC", callback_data="EURAUD=X"),
         InlineKeyboardButton("🇪🇺/🇨🇭 EUR/CHF OTC", callback_data="EURCHF=X")],
        [InlineKeyboardButton("🇳🇿/🇺🇸 NZD/USD OTC", callback_data="NZDUSD=X"),
         InlineKeyboardButton("🇬🇧/🇯🇵 GBP/JPY OTC", callback_data="GBPJPY=X")],
        [InlineKeyboardButton("🇨🇭/🇬🇧 GBP/CHF OTC", callback_data="GBPCHF=X"),
         InlineKeyboardButton("📊 Ver rendimiento histórico", callback_data="ver_rendimiento")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text("📈 Selecciona un par OTC:", reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text("📈 Selecciona un par OTC:", reply_markup=reply_markup)

# ------------------------------
# Selección de par → intervalos
# ------------------------------
async def manejar_seleccion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    par = query.data

    keyboard = [
        [InlineKeyboardButton("1m", callback_data=f"{par}|1m")],
        [InlineKeyboardButton("5m", callback_data=f"{par}|5m")],
        [InlineKeyboardButton("15m", callback_data=f"{par}|15m")],
        [InlineKeyboardButton("1h", callback_data=f"{par}|1h")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text=f"⏱ Selecciona intervalo para {par}:", reply_markup=reply_markup)
# ------------------------------
# Selección de intervalo → señal
# ------------------------------
async def manejar_intervalo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.split("|")
    if len(data) != 2:
        await query.edit_message_text(text=f"⚠️ Error: formato inesperado en {query.data}")
        return

    par, intervalo = data
    await query.edit_message_text(text=f"🔍 Analizando {par} en {intervalo}...")

    modelo, precision, df_hist = entrenar_modelo(par, intervalo)
    senal = generar_senal(par, intervalo, modelo, precision)

    # Teclado con nueva señal y rendimiento histórico
    keyboard = [
        [InlineKeyboardButton("📡 Nueva señal", callback_data="nueva_senal")],
        [InlineKeyboardButton("📊 Ver rendimiento histórico", callback_data=f"ver_rendimiento|{par}|{intervalo}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.user_data["df_hist"] = df_hist

    await context.bot.send_message(chat_id=query.message.chat_id, text=senal, reply_markup=reply_markup)

# ------------------------------
# Nueva señal → volver al menú
# ------------------------------
async def manejar_nueva_senal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await menu_otc(update, context)

# ------------------------------
# Ver rendimiento histórico
# ------------------------------
async def manejar_rendimiento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    parts = query.data.split("|")
    if parts[0] == "ver_rendimiento" and len(parts) == 3:
        _, par, intervalo = parts
        _, _, df_hist = entrenar_modelo(par, intervalo)
    else:
        df_hist = context.user_data.get("df_hist", None)
        par = "PAR DESCONOCIDO"
        intervalo = "INTERVALO"

    if df_hist is None or df_hist.empty:
        await query.edit_message_text("⚠️ No hay datos históricos disponibles para generar el gráfico.")
        return

    buf = generar_grafico_rendimiento(df_hist, par, intervalo)

    await query.message.reply_photo(
        photo=InputFile(buf, filename="rendimiento.png"),
        caption=f"📊 Rendimiento histórico de {par} ({intervalo})"
    )

# ------------------------------
# Configuración del bot (main)
# ------------------------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Comando inicial /start
    app.add_handler(CommandHandler("start", menu_otc))
    # Selección de par (ej: EURUSD=X)
    app.add_handler(CallbackQueryHandler(manejar_seleccion, pattern=r".*=X$"))
    # Selección de intervalo (ej: EURUSD=X|15m)
    app.add_handler(CallbackQueryHandler(manejar_intervalo, pattern=r"^[A-Z]+.*\|[0-9]+[mh]$"))
    # Volver a menú
    app.add_handler(CallbackQueryHandler(manejar_nueva_senal, pattern=r"nueva_senal"))
    # Ver rendimiento histórico (ej: ver_rendimiento|EURUSD=X|15m)
    app.add_handler(CallbackQueryHandler(manejar_rendimiento, pattern=r"^ver_rendimiento.*"))

    app.run_polling()

if __name__ == "__main__":
    main()

<<<<<<< HEAD

=======
>>>>>>> a7e7edd3721ecb31a71f39d6e9f07d77157bea6f
