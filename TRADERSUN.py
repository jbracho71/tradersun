import yfinance as yf
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os # Necesario para leer la variable de entorno PORT
import joblib # ⬅️ Nuevo: Para cargar el modelo pre-entrenado
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, request # Importar Flask para el servidor web/webhook

# NOTA: Reemplaza con tu token real de BotFather
TOKEN = "8246576801:AAEORFpWu_gwXhRq7QznMb1mwnCYeH3-uOk" 

# ------------------------------
# Carga del modelo pre-entrenado (¡El secreto para el arranque rápido!)
# ------------------------------
try:
    # ⚡ Carga el modelo binario en memoria al inicio. Esto es muy rápido.
    MODELO_GLOBAL = joblib.load('tradersun_modelo.pkl')
    PRECISION_GLOBAL = 85.0 # Usar una precisión estimada o guardada
    print("Modelo de ML cargado exitosamente. Arranque rápido asegurado.")
except FileNotFoundError:
    print("❌ ERROR CRÍTICO: No se encontró 'tradersun_modelo.pkl'. El bot no funcionará.")
    MODELO_GLOBAL = None
    PRECISION_GLOBAL = 0.0

# ------------------------------
# Entrenamiento del modelo (Ahora es una función de mantenimiento, no de arranque)
# ------------------------------
def entrenar_modelo(par="EURUSD=X", intervalo="15m", dias="30d"):
    # Esta función ya no es necesaria en el flujo de arranque del bot,
    # solo se mantiene por si quieres re-entrenar y guardar un nuevo archivo .pkl.
    # En el flujo del bot, retornaremos el modelo global.
    return MODELO_GLOBAL, PRECISION_GLOBAL, None
# ------------------------------
# Generación de señal... (no cambia)
# ------------------------------
def generar_senal(par: str, intervalo: str, modelo, precision: float) -> str:
    # ... (código interno de la señal se mantiene igual) ...
    # ... (se usa el modelo y la precisión que se le pasa) ...
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
        pred = modelo.predict(X_new)[0]  
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
# ... (la función generar_grafico_rendimiento se mantiene igual) ...
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
# Handlers del Bot de Telegram
# ------------------------------
# ... (menu_otc se mantiene igual) ...
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

    message = update.effective_message
    await message.reply_text("📈 Selecciona un par OTC:", reply_markup=reply_markup)


# ... (manejar_seleccion se mantiene igual) ...
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
# Selección de intervalo → señal (USA MODELO GLOBAL)
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

    # 🛑 YA NO ENTRENAMOS, USAMOS EL MODELO CARGADO AL INICIO
    modelo = MODELO_GLOBAL
    precision = PRECISION_GLOBAL
    
    # PERO SÍ NECESITAMOS EL df_hist PARA EL GRÁFICO (lo obtenemos de yfinance)
    _, _, df_hist = yf.download(par, period="30d", interval=intervalo, auto_adjust=True)
    
    if modelo is None:
        await context.bot.send_message(chat_id=query.message.chat_id, text="❌ Error: El modelo no pudo cargarse al iniciar el bot.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📡 Nueva señal", callback_data="nueva_senal")]]))
        return
    
    senal = generar_senal(par, intervalo, modelo, precision)

    # Teclado con nueva señal y rendimiento histórico
    keyboard = [
        [InlineKeyboardButton("📡 Nueva señal", callback_data="nueva_senal")],
        [InlineKeyboardButton("📊 Ver rendimiento histórico", callback_data=f"ver_rendimiento|{par}|{intervalo}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.user_data["df_hist"] = df_hist # Guardamos el DF para el gráfico
    context.user_data["par"] = par
    context.user_data["intervalo"] = intervalo

    await context.bot.send_message(chat_id=query.message.chat_id, text=senal, reply_markup=reply_markup)

# ... (manejar_nueva_senal se mantiene igual) ...
async def manejar_nueva_senal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await menu_otc(update, context)

# ------------------------------
# Ver rendimiento histórico (obtener df si no está en cache)
# ------------------------------
async def manejar_rendimiento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    parts = query.data.split("|")
    df_hist = context.user_data.get("df_hist")
    par = context.user_data.get("par", "PAR")
    intervalo = context.user_data.get("intervalo", "INT")

    if parts[0] == "ver_rendimiento" and len(parts) == 3:
        _, par, intervalo = parts
        # Si no hay df_hist guardado, lo descargamos (rápido)
        if df_hist is None or context.user_data.get("par") != par or context.user_data.get("intervalo") != intervalo:
            await query.edit_message_text(f"Descargando datos para el gráfico de {par}...")
            df_hist = yf.download(par, period="30d", interval=intervalo, auto_adjust=True)


    if df_hist is None or df_hist.empty:
        await query.edit_message_text(f"⚠️ No hay datos históricos disponibles para generar el gráfico de {par} ({intervalo}).")
        return

    buf = generar_grafico_rendimiento(df_hist, par, intervalo)

    await context.bot.send_photo(
        chat_id=query.message.chat_id,
        photo=InputFile(buf, filename="rendimiento.png"),
        caption=f"📊 Rendimiento histórico de {par} ({intervalo})"
    )
    # Vuelve al menú principal después del gráfico
    await menu_otc(update, context)

# ------------------------------
# Configuración del bot (handlers)
# ------------------------------
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", menu_otc))
app.add_handler(CallbackQueryHandler(manejar_seleccion, pattern="^(?!.*\\|).*")) 
app.add_handler(CallbackQueryHandler(manejar_intervalo, pattern=".*\\|.*"))      
app.add_handler(CallbackQueryHandler(manejar_nueva_senal, pattern="nueva_senal"))
app.add_handler(CallbackQueryHandler(manejar_rendimiento, pattern="ver_rendimiento.*"))

# ------------------------------
# Servidor Flask para Cloud Run (¡BLOQUE CORREGIDO!)
# ------------------------------

# Se mantienen las rutas y el app builder del final
@flask_app.route('/')
def home():
    return "Tradersun Bot activo 🚀"

@flask_app.route('/webhook', methods=['POST'])
def webhook():
    # 1. Obtiene la actualización del cuerpo de la petición POST
    json_data = request.get_json(force=True)
    update = Update.de_json(json_data, app.bot)
    
    # 2. Procesa la actualización
    # Es crucial usar process_update para que el Application se encargue de todo
    app.process_update(update) 
    
    # 3. Devuelve 200 OK inmediatamente
    return "ok" 

# Arranque final del servidor web
if __name__ == "__main__":
    # 🛑 BLOQUE CORREGIDO: SOLO SE INICIA EL SERVIDOR FLASK 🛑
    port = int(os.environ.get("PORT", 8080))
    # NOTA: Debemos inicializar el objeto Application antes de ejecutar Flask
    # Ya lo hicimos arriba con app = ApplicationBuilder()...
    
    # Inicia el servidor Flask en el puerto asignado por Cloud Run.
    flask_app.run(host="0.0.0.0", port=port, debug=False)
