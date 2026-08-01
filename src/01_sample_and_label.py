# %%
# FAZA 1 — Izdvajanje uzorka korisnika iz auth.txt i obeležavanje target varijable
#
# Iz auth.txt izdvajam podskup korisnika. Svih 104 korisnika
# iz redteam evidencije i 400 nasumičnih korisnika koji se u toj
# evidenciji ne pojavljuju (kontrolni uzorak) i za svaki auth događaj
# određujem vrednost target varijable is_attack na osnovu poklapanja sa
# redteam.txt.
#
# Entitet analize je korisnik, filtriranje i praćenje je po polju dst_user
# (identitet čiji se kredencijali koriste da bi se stiglo do odredišta).

from pathlib import Path
import duckdb
import pandas as pd
import random

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

AUTH = RAW / "auth.txt"
REDTEAM = RAW / "redteam.txt"
REDTEAM_USERS = PROC / "redteam_users.csv"

N_RANDOM = (
    400  # broj nasumičnih korisnika koji se ne pojavljuju u red-team evidenciji napada
)
CAP_PER_USER = 130_000  # gornja granica broja događaja po korisniku
SEED = 7


# %%
# --- 1. Učitavanje redteam korisnika (rezultat Faze 0) ---
rt_users = pd.read_csv(REDTEAM_USERS)["user"].tolist()
print("Redteam korisnika (obavezan deo uzorka):", len(rt_users))

# %%
# --- 2. Izvlačenje svih dst_user vrednosti iz auth.txt ---
# DuckDB skenira fajl direktno sa diska i ne učitava ga u memoriju. Rezultat služi kao populacija iz koje se uzima
# kontrolni (nasumični) deo uzorka.
con = duckdb.connect()
con.execute(f"""
    CREATE VIEW auth AS
    SELECT * FROM read_csv('{AUTH.as_posix()}',
        delim=',', header=false,
        columns={{
            'time':'BIGINT','src_user':'VARCHAR','dst_user':'VARCHAR',
            'src_comp':'VARCHAR','dst_comp':'VARCHAR','auth_type':'VARCHAR',
            'logon_type':'VARCHAR','orientation':'VARCHAR','outcome':'VARCHAR'
        }})
""")

all_users = con.execute("SELECT DISTINCT dst_user FROM auth").df()["dst_user"].tolist()
print("Ukupno različitih dst_user vrednosti u auth.txt:", len(all_users))


# %%
# --- 3. Formiranje uzorka: redteam korisnici + nasumičan kontrolni uzorak ---
# Zadržavaju se svi ljudski nalozi, domenski (U<broj>@DOM<broj>) i lokalni
# (U<broj>@C<broj>), jer redteam korisnici uključuju oba oblika.
# Mašinski nalozi (C###$) i nalozi nepoznatog opsega (U###@?) su
# isključeni jer ne prate ljudski obrazac ponašanja koji se modeluje u
# narednim fazama.
random.seed(SEED)

rt_set = set(rt_users)
kandidati = sorted(
    [
        u
        for u in all_users
        if u not in rt_set
        and u.startswith("U")  # ljudski nalozi
        and "$" not in u  # mašinski nalozi (C###$)
        and not u.endswith("@?")  # nepoznat opseg (U2761@?)
    ]
)

# Kontrola filtera: proverava se šta je odbačeno i da li obrazac U...@DOM...
# obuhvata sve redteam korisnike (oni su po definiciji ljudski nalozi).
kand_set = set(kandidati)
odbaceno = [u for u in all_users if u not in kand_set and u not in rt_set]

print("Ukupno različitih dst_user:", len(all_users))
print("Prošlo filter (kandidati):", len(kandidati))
print("Odbačeno:", len(odbaceno))

print("\nPrimeri ODBAČENIH (prvih 20):")
print(odbaceno[:20])

print("\nPrimeri ZADRŽANIH (prvih 20):")
print(kandidati[:20])

rt_bez_obrasca = [u for u in rt_set if not (u.startswith("U") and "@DOM" in u)]
print("\nRedteam korisnici koji NE odgovaraju obrascu U...@DOM...:", rt_bez_obrasca)

# Kontrolni uzorak se bira iz filtrirane liste kandidata.
random_users = random.sample(kandidati, min(N_RANDOM, len(kandidati)))
uzorak = sorted(rt_set | set(random_users))
print(
    f"\nUzorak: {len(rt_users)} redteam + {len(random_users)} nasumičnih = {len(uzorak)} ukupno"
)

pd.Series(uzorak, name="dst_user").to_csv(PROC / "sampled_users.csv", index=False)

# %%
# --- 4. Filtriranje auth.txt na formirani uzorak, uz ograničenje po korisniku ---
# ROW_NUMBER dodeljuje svakom događaju redni broj unutar korisnika, hronološki
# (od najstarijeg ka najnovijem). Na osnovu tog rednog broja se zadržava samo
# prvih CAP_PER_USER događaja po korisniku, čime se ograničava veličina
# izlaza za korisnike sa izuzetno velikim brojem autentifikacija.
users_sql = ",".join("'" + u.replace("'", "''") + "'" for u in uzorak)

con.execute(f"""
    COPY (
        SELECT time, src_user, dst_user, src_comp, dst_comp,
               auth_type, logon_type, orientation, outcome
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY dst_user ORDER BY time) AS rn
            FROM auth
            WHERE dst_user IN ({users_sql})
        )
        WHERE rn <= {CAP_PER_USER}
    ) TO '{(PROC / "auth_sample.parquet").as_posix()}' (FORMAT parquet)
""")
print("Sačuvano: data/processed/auth_sample.parquet")

# %%
# --- 5. Učitavanje filtriranog uzorka i redteam evidencije ---
auth = pd.read_parquet(PROC / "auth_sample.parquet")
print("Broj redova u uzorku:", len(auth))
print(auth.head())

rt = pd.read_csv(REDTEAM, header=None, names=["time", "user", "src_comp", "dst_comp"])
print("\nUkupno redteam događaja:", len(rt))

# %%
# --- 6. Kreiranje target varijable (is_attack) ---
# Auth red dobija is_attack = 1 ako se poklapa sa redteam događajem po ključu
# (time, dst_user, src_comp, dst_comp); u suprotnom is_attack = 0. LEFT join
# obavezno čuva sve auth redove. Cilj je dodeliti labelu svakom događaju,
# a ne filtrirati skup.
rt_key = rt.rename(columns={"user": "dst_user"})
rt_key["is_attack"] = 1

# Uklanjanje duplikata po ključu sprečava da se auth red umnoži ako u redteam
# evidenciji postoji više zapisa sa istom kombinacijom vremena, korisnika i
# računara.
rt_key = rt_key.drop_duplicates(subset=["time", "dst_user", "src_comp", "dst_comp"])

n_pre = len(auth)

auth = auth.merge(
    rt_key[["time", "dst_user", "src_comp", "dst_comp", "is_attack"]],
    on=["time", "dst_user", "src_comp", "dst_comp"],
    how="left",
)
auth["is_attack"] = auth["is_attack"].fillna(0).astype(int)

# Kontrola: LEFT join ne sme da promeni broj redova.
assert len(auth) == n_pre, f"Merge je promenio broj redova: {n_pre} posle {len(auth)}"
# %%
# --- 7. Provera poklapanja sa redteam evidencijom ---
# Kontrola ispravnosti pristupa: broj auth redova sa is_attack = 1 treba da
# bude blizu ukupnog broja redteam događaja. Potpuno poklapanje (100%) se ne
# očekuje ni pri ispravnoj implementaciji, jer deo redteam događaja nema
# tačan odgovarajući zapis u auth.txt.
oznaceno = int(auth["is_attack"].sum())
print("=" * 50)
print(f"Redteam događaja (očekivano):      {len(rt)}")
print(f"Označeno u uzorku (dst_user join):  {oznaceno}")
print(f"Poklapanje: {oznaceno}/{len(rt)} = {oznaceno / len(rt) * 100:.1f}%")
print("=" * 50)

# Ako je poklapanje ispod 90%, testira se alternativni ključ (src_user).
# Napomena: u ovom skupu podataka src_user i dst_user se u većini redteam
# događaja poklapaju, pa se identičan rezultat za oba ključa ne tumači kao
# dokaz da je izbor ključa nebitan — dst_user ostaje semantički ispravan
# izbor jer predstavlja identitet čiji se kredencijali koriste da bi se
# stiglo do odredišta (redteam.user odgovara upravo tom identitetu).
if oznaceno < 0.9 * len(rt):
    print("\nPoklapanje ispod praga od 90% — testiram alternativni ključ (src_user).")
    tmp = pd.read_parquet(PROC / "auth_sample.parquet")
    rt_src = rt.rename(columns={"user": "src_user"})
    rt_src["hit"] = 1
    tmp = tmp.merge(
        rt_src[["time", "src_user", "src_comp", "dst_comp", "hit"]],
        on=["time", "src_user", "src_comp", "dst_comp"],
        how="left",
    )
    print("Poklapanje preko src_user:", int(tmp["hit"].fillna(0).sum()))

# %%
# --- 8. Čuvanje obeleženog uzorka ---
auth.to_parquet(PROC / "auth_labeled.parquet", index=False)
print("Sačuvano: data/processed/auth_labeled.parquet")
print("\nRaspodela target varijable (is_attack):")
print(auth["is_attack"].value_counts())

# %%
