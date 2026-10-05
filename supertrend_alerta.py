#!/usr/bin/env python3
"""
Alerta Supertrend (US100 / Nasdaq 100, velas de 15 min) -> Discord e/ou ntfy.

Uso:
    python supertrend_alerta.py --teste     # envia uma mensagem de exemplo
    python supertrend_alerta.py             # verifica UMA vez (para o GitHub Actions)
    python supertrend_alerta.py --loop      # fica a correr e verifica a cada 15 min

O webhook e o tópico NUNCA vão no código. Ficam em variáveis de ambiente
(basta definir uma das duas, ou as duas):
    DISCORD_WEBHOOK_URL
    NTFY_TOPICO

ATENÇÃO: os dados gratuitos do Yahoo Finance podem ter atraso (até ~15 min)
e não são idênticos aos do TradingView. Os sinais podem diferir ligeiramente.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
import yfinance as yf

# ----------------------------- Configuração ---------------------------------
TICKER_PADRAO = "NQ=F"      # futuros Nasdaq 100 (quase 24h). Alternativa: "^NDX" (só horário dos EUA)
PERIODO_ATR = 15            # período do ATR (RMA / Wilder)
MULTIPLICADOR = 5.0         # multiplicador do Supertrend
INTERVALO_MIN = 15          # tamanho da vela em minutos
IDADE_MAX_MIN = 45          # ignora sinais de velas mais antigas que isto (evita avisos tardios)
FICHEIRO_ESTADO = Path("estado.json")   # guarda o último sinal enviado (anti-repetição)
FUSO = ZoneInfo("Europe/Lisbon")        # fuso horário mostrado na mensagem


# ----------------------------- Dados ----------------------------------------
def obter_dados(ticker: str):
    """Descarrega velas de 15 min e remove a vela ainda em formação."""
    df = yf.Ticker(ticker).history(period="30d", interval=f"{INTERVALO_MIN}m", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Sem dados para {ticker}.")

    df = df[["High", "Low", "Close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Só avaliamos velas FECHADAS: se a última ainda não terminou, descarta-se.
    agora = datetime.now(timezone.utc)
    if df.index[-1] + timedelta(minutes=INTERVALO_MIN) > agora:
        df = df.iloc[:-1]

    if len(df) < PERIODO_ATR + 5:
        raise RuntimeError("Poucas velas para calcular o indicador.")
    return df


# ----------------------------- Indicador ------------------------------------
def supertrend(df):
    """Replica o Supertrend do script Pine. Devolve (supertrend, close)."""
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)
    p = PERIODO_ATR

    # True Range (na primeira vela é high - low, como no Pine)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    # ATR com RMA (Wilder): começa com a média simples das primeiras p velas
    atr = np.full(n, np.nan)
    atr[p - 1] = tr[:p].mean()
    for i in range(p, n):
        atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p

    src = (h + l) / 2
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    st = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)

    for i in range(p - 1, n):
        up_i = src[i] - MULTIPLICADOR * atr[i]
        dn_i = src[i] + MULTIPLICADOR * atr[i]

        if i > p - 1:
            up1, dn1 = up[i - 1], dn[i - 1]
            if c[i - 1] > up1:
                up_i = max(up_i, up1)
            if c[i - 1] < dn1:
                dn_i = min(dn_i, dn1)

            if trend[i - 1] == -1 and c[i] > dn1:
                trend[i] = 1
            elif trend[i - 1] == 1 and c[i] < up1:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

        up[i], dn[i] = up_i, dn_i
        st[i] = up_i if trend[i] == 1 else dn_i

    return st, c


def detetar_sinal(df):
    """Devolve 'COMPRA', 'VENDA' ou None, olhando só para a última vela fechada."""
    st, c = supertrend(df)
    if np.isnan(st[-2]) or np.isnan(st[-1]):
        return None
    if c[-2] <= st[-2] and c[-1] > st[-1]:   # fecho cruza para cima
        return "COMPRA"
    if c[-2] >= st[-2] and c[-1] < st[-1]:   # fecho cruza para baixo
        return "VENDA"
    return None


# ----------------------------- Estado (anti-repetição) ----------------------
def ler_estado():
    if FICHEIRO_ESTADO.exists():
        try:
            return json.loads(FICHEIRO_ESTADO.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def guardar_estado(chave: str):
    FICHEIRO_ESTADO.write_text(json.dumps({"ultimo_sinal": chave}))


# ----------------------------- Notificações ---------------------------------
def enviar_notificacao(texto: str):
    """Envia para o Discord (webhook) e/ou ntfy, conforme as variáveis definidas."""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    topico = os.environ.get("NTFY_TOPICO")
    if not webhook and not topico:
        sys.exit("Define DISCORD_WEBHOOK_URL e/ou NTFY_TOPICO.")

    if webhook:
        resp = requests.post(
            webhook,
            json={"username": "Alerta US100", "content": texto},
            timeout=20,
        )
        resp.raise_for_status()

    if topico:
        resp = requests.post(
            f"https://ntfy.sh/{topico}",
            data=texto.encode("utf-8"),
            headers={"Title": "Alerta US100", "Priority": "high"},
            timeout=20,
        )
        resp.raise_for_status()


# ----------------------------- Lógica principal -----------------------------
def verificar(ticker: str):
    df = obter_dados(ticker)
    sinal = detetar_sinal(df)
    inicio_vela = df.index[-1]                       # início da última vela fechada
    fim_vela = inicio_vela + timedelta(minutes=INTERVALO_MIN)
    idade_min = (datetime.now(timezone.utc) - fim_vela).total_seconds() / 60

    if sinal is None:
        print(f"[{datetime.now(FUSO):%H:%M}] Sem sinal.")
        return
    if idade_min > IDADE_MAX_MIN:
        print(f"Sinal {sinal} antigo ({idade_min:.0f} min). Ignorado.")
        return

    chave = f"{sinal}|{inicio_vela.isoformat()}"
    if ler_estado().get("ultimo_sinal") == chave:
        print("Sinal já enviado antes. Ignorado.")
        return

    preco = df["Close"].iloc[-1]
    hora = fim_vela.astimezone(FUSO).strftime("%d/%m/%Y %H:%M")
    emoji = "🟢" if sinal == "COMPRA" else "🔴"
    texto = (
        f"{emoji} Sinal de {sinal} - US100 (15m)\n"
        f"Preço de fecho: {preco:.2f}\n"
        f"Hora (Lisboa): {hora}\n"
        f"Fonte: Yahoo Finance ({ticker}), pode ter atraso."
    )
    enviar_notificacao(texto)
    guardar_estado(chave)
    print("Notificação enviada:", sinal)


def dormir_ate_proxima_vela():
    """Espera até ao próximo quarto de hora + 30 s (dá tempo à vela de fechar nos dados)."""
    agora = datetime.now(timezone.utc)
    minutos_a_somar = INTERVALO_MIN - (agora.minute % INTERVALO_MIN)
    proxima = (agora + timedelta(minutes=minutos_a_somar)).replace(second=30, microsecond=0)
    time.sleep(max((proxima - agora).total_seconds(), 1))


def main():
    ap = argparse.ArgumentParser(description="Alerta Supertrend -> Discord/ntfy")
    ap.add_argument("--teste", action="store_true", help="envia uma mensagem de exemplo")
    ap.add_argument("--loop", action="store_true", help="corre continuamente (a cada 15 min)")
    ap.add_argument("--ticker", default=TICKER_PADRAO, help="NQ=F (padrão) ou ^NDX")
    args = ap.parse_args()

    if args.teste:
        enviar_notificacao("✅ Teste: o alerta do US100 está a funcionar.")
        print("Mensagem de teste enviada.")
        return

    if args.loop:
        while True:
            try:
                verificar(args.ticker)
            except Exception as e:  # não deixa o loop morrer por um erro de rede
                print("Erro:", e)
            dormir_ate_proxima_vela()
    else:
        verificar(args.ticker)


if __name__ == "__main__":
    main()
