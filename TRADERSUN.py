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

TOKEN = "8246576801:AAEORFpWu_gwXhRq7QznMb1mwnCYeH3-uOk"  # Reemplaza con tu token real de BotFather

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
# Mensaje estilo Trader Loco
# ------------------------------
def generar_mensaje_estilo_trader(par, intervalo, rsi, cci, stoch, adx, pred, precision, confianza, atr_index):
    tendencia = "alcista" if pred == 1 else "bajista"
    tipo = "🟢 CALL (COMPRAR)" if pred == 1 else "🔴 PUT (VENDER)"

    prob_entrada = round(precision, 2)
    prob_reversion = round(100 - precision, 2)

    mensaje = (
        f"🧠 Analizando indicadores para {par} OTC\n"
        f"📉 Tendencia detectada: {tendencia}\n\n"
        f"📊 Marco de tiempo: {intervalo}\n"
        f"{tipo}\n\n"
        f"📈 RSI={rsi:.2f}, CCI={cci:.2f}, STOCH={stoch:.2f}, ADX={adx:.2f}\n"
        f"📊 Volatilidad (ATR): {atr_index:.2f}/100\n"
        f"✅ Probabilidad de entrada exitosa: {prob_entrada}%\n"
        f"🔄 Probabilidad de reversión: {prob_reversion}%\n"
        f"📊 Nivel de confianza (0–100): {confianza:.2f}\n\n"
        f"⚠️ No operar fuera de esta señal"
    )
    return mensaje
# ------------------------------
# Generación de señal
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

        # Convertir explícitamente a float para evitar errores de Series
        rsi = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])
        cci = float(ta.trend.CCIIndicator(high, low, close).cci().iloc[-1])
        stoch = float(ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1])
        adx = float(ta.trend.ADXIndicator(high, low, close).adx().iloc[-1])

        atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
        atr_index = (atr / float(df["High"].max())) * 100

        if adx < 15:
            return (
                f"⚠️ Mercado plano detectado para {par} ({intervalo}).\n"
                f"ADX={adx:.2f} < 15 → evitar operar.\n"
                f"RSI={rsi:.2f}, CCI={cci:.2f}, STOCH={stoch:.2f}, ATR={atr_index:.2f}/100"
            )

        X_new = pd.DataFrame([[rsi, cci, stoch, adx]], columns=["RSI", "CCI", "STOCH", "ADX"])
        pred = modelo.predict(X_new)[0]
        confianza = float(modelo.predict_proba(X_new)[0][pred] * 100)

        return generar_mensaje_estilo_trader(par, intervalo, rsi, cci, stoch, adx, pred, precision, confianza, atr_index)

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

    # Teclado solo con nueva señal y rendimiento histórico
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
    if len(parts) == 3:
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
# Configuración del bot
# ------------------------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Comando inicial /start
    app.add_handler(CommandHandler("start", menu_otc))
    app.add_handler(CallbackQueryHandler(manejar_seleccion, pattern=r".*=X$"))
    app.add_handler(CallbackQueryHandler(manejar_intervalo, pattern=r".*\|.*"))
    app.add_handler(CallbackQueryHandler(manejar_nueva_senal, pattern=r"nueva_senal"))
    app.add_handler(CallbackQueryHandler(manejar_rendimiento, pattern=r"ver_rendimiento.*"))

    app.run_polling()

if __name__ == "__main__":
    main()
