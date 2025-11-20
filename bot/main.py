import os
from typing import Optional

import httpx
from discord import Color, Embed, Intents, app_commands
from discord.ext import commands
from dotenv import load_dotenv


load_dotenv(".env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_SYMBOL = os.getenv("BYBIT_TEST_SYMBOL", "ETH/USDT:USDT")
DEFAULT_SL = float(os.getenv("DEFAULT_SL_PCT", "1.0"))

intents = Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


def build_embed(signal: dict, symbol: str, sl_percentage: float) -> Embed:
    decision = signal.get("decision", "NEUTRAL")
    color = Color.light_grey()
    if decision == "LONG":
        color = Color.green()
    elif decision == "SHORT":
        color = Color.red()

    entry = signal.get("entry_price")
    sl_price = signal.get("sl_price")
    tp_price = signal.get("tp_price")
    rr = signal.get("risk_reward_ratio")
    conf = signal.get("confidence_score")
    reasoning = signal.get("reasoning") or "No reasoning provided."

    sl_pct_display: Optional[float] = None
    tp_pct_display: Optional[float] = None
    if entry and sl_price:
        sl_pct_display = ((entry - sl_price) / entry) * 100 if decision == "LONG" else ((sl_price - entry) / entry) * 100
    if entry and tp_price:
        tp_pct_display = ((tp_price - entry) / entry) * 100 if decision == "LONG" else ((entry - tp_price) / entry) * 100

    embed = Embed(title=f"Signal Analysis: {symbol}", color=color)
    embed.add_field(name="Decision", value=decision, inline=True)
    embed.add_field(name="Confidence", value=f"{conf:.1f}%" if conf is not None else "N/A", inline=True)
    embed.add_field(name="R:R", value=f"1:{rr:.2f}" if rr else "N/A", inline=True)

    embed.add_field(name="Entry", value=f"{entry:.4f}" if entry else "N/A", inline=True)
    embed.add_field(
        name="Stop Loss",
        value=f"{sl_price:.4f} ({-sl_pct_display:.2f}%)" if sl_price and sl_pct_display is not None else f"Fixed {-sl_percentage}%",
        inline=True,
    )
    embed.add_field(
        name="Take Profit",
        value=f"{tp_price:.4f} (+{tp_pct_display:.2f}%)" if tp_price and tp_pct_display is not None else "N/A",
        inline=True,
    )

    embed.add_field(name="Analysis", value=reasoning, inline=False)
    embed.set_footer(text="This is not financial advice. Trade at your own risk.")
    return embed


@bot.tree.command(name="position", description="Fetch a futures trading signal")
@app_commands.describe(
    symbol="Trading pair, e.g., ETH/USDT:USDT",
    sl="Stop loss percent (e.g., 1.0 for 1%)",
)
async def position(interaction, symbol: str = DEFAULT_SYMBOL, sl: float = DEFAULT_SL):
    await interaction.response.defer(thinking=True)
    url = f"{API_BASE}/signal"
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(url, json={"symbol": symbol, "sl_percentage": sl})
            resp.raise_for_status()
        except Exception as exc:
            await interaction.followup.send(f"Failed to fetch signal: {exc}")
            return

    data = resp.json()
    embed = build_embed(data, symbol=symbol, sl_percentage=sl)
    await interaction.followup.send(embed=embed)


def main():
    if not BOT_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN is not set in .env")
    @bot.event
    async def on_ready():
        synced = await bot.tree.sync()
        print(f"Logged in as {bot.user} | Synced {len(synced)} commands")
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
