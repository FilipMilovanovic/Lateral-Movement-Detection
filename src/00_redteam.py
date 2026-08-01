# %%
# FAZA 0 — Istraživanje redteam.txt (ground truth napada)
#
# Identifikujem entitete (korisnike i računare) koji učestvuju u napadima i
# analiziram raspored napada kroz 58 dana prikupljanja podataka.
# Rezultati direktno određuju obavezan sastav uzorka u Fazi 1 i
# granicu train/valid/test podele u Fazi 2.

# Radi se isključivo nad redteam.txt, auth.txt se u ovoj fazi ne učitava.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
REDTEAM = RAW / "redteam.txt"

SEC_PER_DAY = 86400  # rezolucija vremena je 1 sekunda; vreme kreće od 1

# %%
# --- 1. Učitavanje redteam.txt ---
# Fajl je bez zaglavlja, kolone: time, user@domain, src_comp, dst_comp.
cols = ["time", "user", "src_comp", "dst_comp"]
rt = pd.read_csv(REDTEAM, header=None, names=cols)

print("Ukupno redteam događaja:", len(rt))
print("\nPrvih 5 redova:")
print(rt.head())

# %%
# --- 2. Izvedene kolone: dan i sat u danu ---
# Osnova za analizu vremenskog rasporeda napada u narednim koracima.
rt["day"] = (rt["time"] // SEC_PER_DAY) + 1  # dan 1..58
rt["hour_of_day"] = (rt["time"] % SEC_PER_DAY) // 3600  # sat u danu 0..23

print("Najraniji trenutak:", rt["time"].min(), "| dan:", int(rt["day"].min()))
print("Najkasniji trenutak:", rt["time"].max(), "| dan:", int(rt["day"].max()))
print("Raspon u danima:", int(rt["day"].min()), "-", int(rt["day"].max()))

# %%
# --- 3. Entiteti koji učestvuju u napadima ---
# Korisnici se obavezno uključuju u uzorak formiran u Fazi 1, jer predstavljaju
# jedine dostupne pozitivne primere za nadgledano učenje. Računari se ovde
# samo dokumentuju i čuvaju kao referenca.
users = sorted(rt["user"].unique())
src_comps = sorted(rt["src_comp"].unique())
dst_comps = sorted(rt["dst_comp"].unique())
all_comps = sorted(set(src_comps) | set(dst_comps))

print("Različitih korisnika u napadima:", len(users))
print("Različitih izvornih računara:", len(src_comps))
print("Različitih odredišnih računara:", len(dst_comps))
print("Ukupno različitih računara (izvor ∪ odredište):", len(all_comps))

print("\nKorisnici (kompromitovani nalozi):")
print(users)

# %%
# --- 4. Raspored napada po danima ---
# Neophodno za postavljanje vremenske train/test granice u Fazi 2: nasumična
# ili ravnomerna podela nije prihvatljiva ako su napadi koncentrisani u
# pojedinim danima.
by_day = rt.groupby("day").size()
print("Broj napada po danu:")
print(by_day)

print("\nDana sa bar jednim napadom:", rt["day"].nunique(), "od 58")

# %%
# --- 5. Vizuelizacija rasporeda napada kroz vreme ---


RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4))
by_day.reindex(range(1, 59), fill_value=0).plot(kind="bar", ax=ax)
ax.set_xlabel("Dan (1–58)")
ax.set_ylabel("Broj redteam događaja")
ax.set_title("Raspored napada kroz vreme")
plt.tight_layout()
plt.savefig(RESULTS / "redteam_by_day.png", dpi=120)
print("Grafik sačuvan u results/redteam_by_day.png")

# %%
# --- 6. Čuvanje liste entiteta  ---
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

pd.Series(users, name="user").to_csv(PROC / "redteam_users.csv", index=False)
pd.Series(all_comps, name="computer").to_csv(
    PROC / "redteam_computers.csv", index=False
)
print("Sačuvano: data/processed/redteam_users.csv i redteam_computers.csv")

# %%
