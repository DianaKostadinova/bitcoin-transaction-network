"""
Final report generator: compiles all phases (structural, centrality,
community, temporal, anomaly analysis) into one polished PDF with embedded
charts, tables and a final verdict section.

Output: docs/Bitcoin_Transaction_Network_Report.pdf
"""

from pathlib import Path
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
OUT_PDF = DOCS / "Bitcoin_Transaction_Network_Report.pdf"

# --- Cyrillic-capable fonts (base14 PDF fonts have no Cyrillic glyphs) ---
FONTS_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS_DIR / "arialbd.ttf")))

styles = getSampleStyleSheet()
for s in styles.byName.values():
    s.fontName = "Arial"

styles.add(ParagraphStyle(name="TitleMK", fontName="Arial-Bold", fontSize=24,
                           alignment=TA_CENTER, spaceAfter=10, leading=28))
styles.add(ParagraphStyle(name="SubtitleMK", fontName="Arial", fontSize=13,
                           alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                           spaceAfter=6))
styles.add(ParagraphStyle(name="H1", fontName="Arial-Bold", fontSize=17,
                           spaceBefore=18, spaceAfter=10,
                           textColor=colors.HexColor("#1f3a5f")))
styles.add(ParagraphStyle(name="H2", fontName="Arial-Bold", fontSize=13,
                           spaceBefore=12, spaceAfter=6,
                           textColor=colors.HexColor("#2c5282")))
styles.add(ParagraphStyle(name="BodyMK", fontName="Arial", fontSize=10.2,
                           leading=14.5, alignment=TA_JUSTIFY, spaceAfter=8))
styles.add(ParagraphStyle(name="Caption", fontName="Arial", fontSize=8.5,
                           alignment=TA_CENTER, textColor=colors.HexColor("#666666"),
                           spaceAfter=14))
styles.add(ParagraphStyle(name="Verdict", fontName="Arial", fontSize=11,
                           leading=16, alignment=TA_JUSTIFY, spaceAfter=10))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def body(text):
    story.append(Paragraph(text, styles["BodyMK"]))


def caption(text):
    story.append(Paragraph(text, styles["Caption"]))


def hrule():
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc"),
                             spaceBefore=4, spaceAfter=10))


def add_image(path: Path, max_width=6.3 * inch, max_height=4.6 * inch):
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = min(max_width / w, max_height / h)
    story.append(Image(str(path), width=w * ratio, height=h * ratio))


def df_to_table(df: pd.DataFrame, col_widths=None, font_size=7.6):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Arial"),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)


def fmt(df, cols, decimals=None):
    out = df[cols].copy()
    if decimals:
        for c, d in decimals.items():
            if c in out.columns:
                out[c] = out[c].astype(float).round(d)
    return out


# ============================================================ TITLE PAGE
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Анализа на Bitcoin Transaction Network", styles["TitleMK"]))
story.append(Paragraph("Graph-based анализа на структурата, централноста, заедниците, "
                        "временската динамика и аномалиите во Bitcoin trust-мрежите",
                        styles["SubtitleMK"]))
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("Датасети: soc-sign-bitcoinotc, soc-sign-bitcoinalpha (SNAP, Stanford)",
                        styles["SubtitleMK"]))
story.append(Paragraph("Септември 2026", styles["SubtitleMK"]))
story.append(Spacer(1, 1.8 * inch))
story.append(Paragraph(
    "Финален извештај — генериран автоматски врз основа на податочна обработка, "
    "граф-алгоритми (NetworkX, python-louvain) и статистичка анализа.",
    styles["Caption"]))
story.append(PageBreak())

# ============================================================ TOC / INTRO
h1("1. Вовед и опис на податоците")
body(
    "Овој извештај ја сумира комплетната анализа на Bitcoin трансакциска мрежа "
    "претставена како граф, во кој јазлите се кориснички/Bitcoin адреси, а врските "
    "претставуваат трансакциски односи (оценки на доверба) помеѓу нив."
)

h2("1.1 Што точно претставува датасетот")
body(
    "Bitcoin трансакциите се анонимни и неповратни, па платформите за P2P тргување "
    "воведоа <b>меѓусебно оценување на доверба</b>: по секоја трговска интеракција, "
    "корисникот го оценува соговорникот од <b>-10 (целосна недоверба) до +10 "
    "(целосна доверба)</b>. Токму овие оценки го формираат нашиот граф: "
    "<b>јазол</b> = анонимизирана адреса, <b>врска</b> = чин на оценување "
    "(насочена, со тежина = рејтинг и временска ознака)."
)
body(
    "<b>Важна напомена:</b> ова не се сурови blockchain трансакции туку "
    "<i>who-trusts-whom</i> мрежи на доверба — стандарден proxy во истражувачката "
    "литература, бидејќи целосниот on-chain граф е непрактичен за локална анализа. "
    "Заклучоците важат за структурата на довербата, не за монетарни износи."
)

h2("1.2 Разлика помеѓу Bitcoin OTC и Bitcoin Alpha")
body(
    "Двата датасети доаѓаат од <b>две одделни платформи</b> со ист формат, но "
    "различни заедници корисници. Ги анализираме и двете за да провериме дали "
    "наодите се случајност или реален образец — ако истиот резултат излезе кај "
    "двете, заклучокот е поробустен."
)
otc_alpha_rows = pd.DataFrame({
    "Својство": ["Јазли (адреси)", "Врски (оценки)", "Просечен out-degree",
                 "Густина", "Модуларност на заедници"],
    "Bitcoin OTC": ["5,881", "35,592", "6.05", "0.00103", "0.488 (18 заедници)"],
    "Bitcoin Alpha": ["3,783", "24,186", "6.39", "0.00169", "0.471 (21 заедница)"],
})
df_to_table(otc_alpha_rows, font_size=8)
story.append(Spacer(1, 8))
body(
    "OTC е поголема мрежа, Alpha е погуста (повисок просечен степен и покрај "
    "помалку јазли). И покрај тоа, структурните образци (small-world, доминантен "
    "hub, слична community-структура) се речиси идентични во двете — знак дека "
    "наодите не се артефакт на еден датасет."
)

body(
    "Анализата е спроведена низ 8 фази, секоја со јасна цел: (1) подготовка на "
    "податоци, (2) структура на мрежата, (3) кои адреси се најактивни/централни, "
    "(4) заедници на доверба, (5) временска динамика, (6) детекција на сомнителни "
    "адреси, (7) длабинска signed-network анализа. Секоја фаза подолу вели "
    "<i>што прави скриптата</i>, <i>зошто</i> и <i>што најдовме</i>."
)

# ============================================================ PHASE 2
story.append(PageBreak())
h1("2. Структурна анализа на мрежата")
struct_df = pd.read_csv(RESULTS / "structural_summary.csv")

body(
    "<b>Цел:</b> да се измери „обликот“ на мрежата — колку е поврзана, колку "
    "брзо доверба/влијание може да патува низ неа, и дали личи на реален "
    "социјален граф. Скриптата <code>src/structural_analysis.py</code> пресметува "
    "густина, reciprocity, clustering, поврзани компоненти, дијаметар и просечен "
    "пат."
)
disp = struct_df.copy()
disp["density"] = disp["density"].round(5)
disp["reciprocity"] = disp["reciprocity"].round(3)
disp["avg_clustering_undirected"] = disp["avg_clustering_undirected"].round(3)
disp["largest_wcc_fraction"] = (disp["largest_wcc_fraction"] * 100).round(1)
disp["avg_shortest_path_largest_wcc"] = disp["avg_shortest_path_largest_wcc"].round(2)
disp = disp.rename(columns={
    "dataset": "Датасет", "nodes": "Јазли", "edges": "Врски", "density": "Густина",
    "reciprocity": "Reciprocity", "avg_clustering_undirected": "Clustering",
    "weakly_connected_components": "WCC", "strongly_connected_components": "SCC",
    "largest_wcc_size": "Largest WCC (n)", "largest_wcc_fraction": "Largest WCC (%)",
    "diameter_largest_wcc": "Дијаметар", "avg_shortest_path_largest_wcc": "Просечен пат",
})
df_to_table(disp[["Датасет", "Јазли", "Врски", "Густина", "Reciprocity", "Clustering",
                   "WCC", "SCC", "Largest WCC (%)", "Дијаметар", "Просечен пат"]],
            font_size=7.0)
story.append(Spacer(1, 10))

h2("Наоди")
body(
    "И двете мрежи се <b>small-world</b>: мал дијаметар (9-10) и просечен пат од "
    "само ~3.6 чекори и покрај илјадници јазли — доверба „патува“ низ речиси "
    "целата мрежа за 3-4 чекори. Reciprocity е висок (0.79 / 0.83, довербата "
    "често е взаемна), но многуте мали силно-поврзани компоненти (1,144 / 540) "
    "покажуваат дека вистински „кружни“ циклуси на доверба меѓу повеќе страни се "
    "ретки. Двете мрежи имаат по еден доминантен гигант-компонент (>99.8% од "
    "јазлите) — типично за реални социјални мрежи."
)
story.append(Spacer(1, 6))
add_image(RESULTS / "degree_distribution_bitcoin_otc.png")
caption("Слика 1. Дистрибуција на in/out-degree — Bitcoin OTC (log-скала). "
        "Тежок реп: мал број адреси со стотици врски, огромно мнозинство со <10.")
add_image(RESULTS / "degree_distribution_bitcoin_alpha.png")
caption("Слика 2. Дистрибуција на in/out-degree — Bitcoin Alpha. Истиот "
        "хеви-тејл образец, потврдувајќи scale-free/hub-driven структура.")

# ============================================================ PHASE 3
story.append(PageBreak())
h1("3. Централност — најактивни и најцентрални адреси")
body(
    "<b>Цел:</b> да се одреди кои адреси играат клучна улога — по активност, по "
    "влијателност, или како мостови помеѓу делови од мрежата. Скриптата "
    "<code>src/centrality_analysis.py</code> пресметува 4 метрики:"
)
body(
    "• <b>In/Out-degree</b> — колку различни адреси примиле/дале оценка. "
    "Пример: јазол <b>35</b> во OTC има in=535, out=763 — ~88x над просекот "
    "(avg_in_degree=6.05). Не зема предвид <i>кој</i> оценува.<br/>"
    "• <b>PageRank</b> — истиот алгоритам како Google: важен си ако те поврзуваат "
    "важни адреси, не само многу адреси.<br/>"
    "• <b>Betweenness</b> — колку пати јазолот лежи на најкраткиот пат помеѓу "
    "други два јазли (мост помеѓу делови од мрежата); апроксимирано (k=500 "
    "sample) заради брзина.<br/>"
    "• <b>HITS (hub/authority)</b> — authority = кон него сочат добри hub-ови "
    "(веродостоен); hub = тој сочи кон добри authorities. За разлика од "
    "PageRank, раздвојува „кому му веруваат“ од „кој верува во добри луѓе“."
)
body(
    "<i>Методолошка забелешка:</i> ова е класична graph theory, не deep "
    "learning — детерминистички пресметки врз графот, не невронски мрежи/"
    "embeddings (тоа би било node2vec/GNN пристап, непотребен за граф со "
    "илјадници јазли)."
)

h2("Резултати по датасет")
for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    top_df = pd.read_csv(RESULTS / f"top_addresses_{name}.csv")
    h2(f"{label} — топ 8 по клучни метрики")
    sub = top_df.head(8)[["rank", "in_degree_node", "in_degree_value",
                           "pagerank_node", "pagerank_value",
                           "betweenness_node", "betweenness_value"]].copy()
    sub["pagerank_value"] = sub["pagerank_value"].round(5)
    sub["betweenness_value"] = sub["betweenness_value"].round(5)
    sub.columns = ["Ранг", "Јазол (in-deg)", "In-degree", "Јазол (PageRank)",
                    "PageRank", "Јазол (betweenness)", "Betweenness"]
    df_to_table(sub, font_size=7.2)
    story.append(Spacer(1, 10))

h2("Наоди")
body(
    "Во <b>Bitcoin OTC</b>, јазол <b>35</b> доминира по сите метрики "
    "(in=535, out=763, највисок PageRank и betweenness) — класичен "
    "hub/маркет-мејкер. Јазли 2642, 1810, 905, 2028 постојано се повторуваат во "
    "топ-листите — мала група „супер-поврзани“ адреси. Во <b>Bitcoin Alpha</b>, "
    "истиот образец: јазол <b>1</b> доминира (in=398, out=490). HITS "
    "hub/authority топ-листите (results/top_addresses_*.csv) содржат <i>поинакви</i> "
    "јазли од degree/PageRank — доказ дека „активност“ и „структурна важност“ не "
    "се секогаш исти адреси."
)

# ============================================================ PHASE 4
story.append(PageBreak())
h1("4. Детекција на заедници (Community Detection)")
comm_overview = pd.read_csv(RESULTS / "community_overview.csv")
body(
    "<b>Цел:</b> да се откријат групи адреси кои оценуваат/тргуваат многу повеќе "
    "меѓу себе отколку со остатокот од мрежата — природни „кругови на доверба“. "
    "Скриптата <code>src/community_detection.py</code> го користи Louvain "
    "алгоритмот: групите се бараат оптимизациски (не со тренирање), преместувајќи "
    "јазли во соседна група секогаш кога тоа ја подобрува <b>модуларноста</b> — "
    "мерка (-1 до +1) колку повеќе врски има <i>внатре</i> во групите отколку "
    "случајно (>0.3 = реална структура, не артефакт). Пресметано врз "
    "нетежинскиот граф, бидејќи Louvain бара не-негативни тежини; знакот на "
    "довербата е анализиран одделно по заедница (колона „Просечен рејтинг“)."
)
ov = comm_overview.copy()
ov["modularity"] = ov["modularity"].round(3)
ov["largest_community_fraction"] = (ov["largest_community_fraction"] * 100).round(1)
ov.columns = ["Датасет", "Број заедници", "Модуларност", "Најголема заедница (n)",
              "Најголема заедница (%)"]
df_to_table(ov, font_size=8)
story.append(Spacer(1, 10))

body(
    "Модуларност 0.488 (OTC, 18 заедници) и 0.471 (Alpha, 21) — силна, реална "
    "структура. Ниту една заедница не доминира (максимум 21-24% од мрежата): "
    "секоја мрежа е федерација на споредливо-големи довербени кластери, не еден "
    "монолитен блок."
)

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    cs = pd.read_csv(RESULTS / f"community_summary_{name}.csv").head(5)
    h2(f"{label} — топ 5 најголеми заедници")
    cs2 = cs.copy()
    cs2["internal_density"] = cs2["internal_density"].round(4)
    cs2["avg_pagerank"] = cs2["avg_pagerank"].round(5)
    cs2["avg_rating_of_members"] = cs2["avg_rating_of_members"].round(2)
    cs2.columns = ["ID", "Големина", "Внатрешни врски", "Густина", "Просечен PageRank",
                    "Просечен рејтинг"]
    df_to_table(cs2, font_size=7.4)
    story.append(Spacer(1, 8))

add_image(RESULTS / "network_communities_bitcoin_otc.png", max_height=5.2 * inch)
caption("Слика 3. Bitcoin OTC — мрежа обоена по Louvain заедница (largest component, "
        "топ 2000 јазли по degree, модуларност=0.488). Јасно се гледаат "
        "визуелно-одвоени кластери на доверба.")
add_image(RESULTS / "network_communities_bitcoin_alpha.png", max_height=5.2 * inch)
caption("Слика 4. Bitcoin Alpha — истата визуелизација (модуларност=0.471).")

# ============================================================ PHASE 5
story.append(PageBreak())
h1("5. Временска (temporal) анализа")
body(
    "<b>Цел:</b> дали мрежата расте рамномерно или во бранови, и дали денешните "
    "најцентрални адреси биле важни од почеток или постепено „пораснале“. "
    "Скриптата <code>src/temporal_analysis.py</code> ги следи растот, неделната "
    "активност (со z-score детекција на избувнувања) и PageRank-рангот на топ-5 "
    "јазли низ 10 временски снимки од историјата."
)

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    add_image(RESULTS / f"growth_{name}.png")
    caption(f"Раст на {label} низ време — кумулативен број врски (сино) и јазли (црвено).")

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    add_image(RESULTS / f"activity_{name}.png", max_height=5.0 * inch)
    caption(f"{label} — месечна активност (горе) и неделна позитивна/негативна "
            f"поделба на оценки (долу).")

body(
    "<i>Z-score</i> (не ML): z = (број_врски_таа_недела − просек) / стд.девијација. "
    "Z>2.5 значи активност статистички невообичаено над нормалата — сметаме ја за "
    "„избувнување“ (burst)."
)

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    bursts = pd.read_csv(RESULTS / f"bursts_{name}.csv")
    h2(f"{label} — детектирани „burst“ недели (z-score > 2.5)")
    b = bursts.copy()
    b["z_score"] = b["z_score"].round(2)
    b.columns = ["Недела", "Број оценки", "Z-score"]
    df_to_table(b, font_size=7.6)
    story.append(Spacer(1, 8))

body(
    "Најизразениот burst во двете мрежи е неделата од <b>2011-06-05/12</b>, со "
    "980 (OTC) и 974 (Alpha) оценки во една недела — 6-9x над просекот (z-score "
    "6.5 / 8.7). Bitcoin OTC има дополнителни, послаби, но сепак значајни избувнувања "
    "во 2013 (април и август), што се совпаѓа со познатите периоди на нагло "
    "зголемен интерес за Bitcoin во тие месеци."
)

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    add_image(RESULTS / f"centrality_evolution_{name}.png")
    caption(f"{label} — PageRank ранг (1=највисок) на денешните топ-5 јазли, "
            f"следен низ 10 растечки временски снимки. Опаѓачка линија = адресата "
            f"стекнувала централност постепено; рамна ниска линија = централна "
            f"од самиот почеток.")

body(
    "Еволуцијата на централноста покажува дека доминантните јазли (35 за OTC, 1 за "
    "Alpha) биле меѓу најцентралните веќе во раните снимки на мрежата — тие не се "
    "„пораснале“ во влијателност дополнително, туку веднаш стекнале голем број врски, "
    "што е конзистентно со рана регистрација и/или улога на маркет-мејкер на "
    "платформата."
)

# ============================================================ PHASE 6
story.append(PageBreak())
h1("6. Детекција на аномалии / сомнителни адреси")
body(
    "<b>Цел:</b> да се флагираат адреси кои заслужуваат понатамошна проверка. "
    "Скриптата <code>src/anomaly_detection.py</code> применува две прости, "
    "објаснливи хевристики: (а) <b>концентрација на негативни оценки</b> — ≥5 "
    "примени оценки, ≥50% негативни (widespread distrust, не еден спор), и (б) "
    "<b>bursty rating</b> — ≥15 дадени оценки во еден ден (сигнал за sybil/бот "
    "однесување)."
)
add_image(RESULTS / "rating_sign_distribution_bitcoin_otc.png", max_width=4.2 * inch, max_height=3.4 * inch)
add_image(RESULTS / "rating_sign_distribution_bitcoin_alpha.png", max_width=4.2 * inch, max_height=3.4 * inch)
caption("Слика 5-6. Дистрибуција на знак на оценки — двете мрежи се доминантно "
        "позитивни (>85% позитивни оценки), што е очекувано за trust-мрежи каде "
        "негативните оценки носат репутациски трошок за оценувачот.")

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    neg = pd.read_csv(RESULTS / f"anomalies_negative_{name}.csv")
    h2(f"{label} — топ 8 адреси со концентрирани негативни оценки "
       f"(вкупно флагирани: {len(neg)})")
    n = neg.head(8)[["node", "num_ratings_received", "num_distinct_raters",
                      "avg_rating", "neg_fraction"]].copy()
    n["avg_rating"] = n["avg_rating"].round(2)
    n["neg_fraction"] = (n["neg_fraction"] * 100).round(0).astype(int)
    n.columns = ["Јазол", "Примени оценки", "Различни оценувачи", "Просечен рејтинг",
                 "% негативни"]
    df_to_table(n, font_size=7.6)
    story.append(Spacer(1, 8))

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    burst_r = pd.read_csv(RESULTS / f"anomalies_bursty_raters_{name}.csv")
    h2(f"{label} — топ 5 „bursty rater“ настани "
       f"(вкупно флагирани: {len(burst_r)})")
    br = burst_r.head(5).copy()
    br.columns = ["Јазол", "Датум", "Оценки во тој ден"]
    df_to_table(br, font_size=7.6)
    story.append(Spacer(1, 8))

body(
    "Флагираните адреси се <b>кандидати за истрага</b>, не потврдена измама. "
    "Јазол 4747 (OTC): 14 негативни оценки од 14 различни оценувачи (100% "
    "негативни) — силен сигнал за конзистентно distrusted актер. Групите соседни "
    "ID-броеви со идентичен bursty-образец на ист датум (пр. 3786-3795, "
    "2013-08-15) укажуваат на можна координирана/sybil регистрација на сметки."
)

# ============================================================ PHASE 8
story.append(PageBreak())
h1("7. Длабинска анализа на signed-мрежа: Fairness & Goodness и Structural Balance")
body(
    "<b>Цел:</b> Фазите 2-6 работат на секаков граф; овде искористивме дека "
    "врските носат <b>знак</b> (-10..+10) за да провериме дали Фаза 6 (проста "
    "хевристика) се потврдува со посложен метод, и дали довербата во мрежата "
    "личи на реална социјална структура. Скрипта: "
    "<code>src/signed_network_analysis.py</code>."
)

h2("7.1 Fairness & Goodness")
body(
    "<b>Проблем со Фаза 6:</b> просечен рејтинг третира еднакво оценка од "
    "докажан трговец и од сомнителна нова сметка. Скриптата итеративно "
    "пресметува <b>Goodness(v) ∈ [-1,+1]</b> (колку е веродостоен v, пондерирано "
    "со тоа колку се доверливи оценувачите) и <b>Fairness(u) ∈ [0,1]</b> (колку "
    "се веродостојни оценките на u). <i>Ограничување:</i> адреси со само 1 "
    "примена оценка автоматски добиваат +1/-1 (мал примерок, не сигнал за "
    "измама) — затоа пресметавме и филтрирана листа (≥5 оценки, исто прагче "
    "како Фаза 6)."
)

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    add_image(RESULTS / f"fairness_goodness_hist_{name}.png", max_width=5.2 * inch, max_height=3.4 * inch)
    caption(f"{label} — дистрибуција на Goodness резултати. Повеќето адреси се "
            f"близу +1 (доверливи), потврдувајќи дека мрежата е доминантно "
            f"позитивна (конзистентно со Фаза 6).")

for name, label in [("bitcoin_otc", "Bitcoin OTC"), ("bitcoin_alpha", "Bitcoin Alpha")]:
    rel = pd.read_csv(RESULTS / f"fairness_goodness_reliable_{name}.csv").sort_values("goodness").head(6)
    h2(f"{label} — топ 6 најнедоверливи адреси (≥5 примени оценки)")
    r = rel[["node", "fairness", "goodness", "num_ratings_received"]].copy()
    r["fairness"] = r["fairness"].round(3)
    r["goodness"] = r["goodness"].round(3)
    r.columns = ["Јазол", "Fairness", "Goodness", "Примени оценки"]
    df_to_table(r, font_size=7.8)
    story.append(Spacer(1, 8))

overlap_df = pd.read_csv(RESULTS / "signed_analysis_overview.csv")
body(
    "<b>Крос-валидација со Фаза 6:</b> преклопувањето помеѓу најниска-goodness "
    "адреси (≥5 оценки) и адресите веќе флагирани во Фаза 6 е <b>91.1%</b> за "
    "OTC (144/158) и <b>92.2%</b> за Alpha (47/51). Два методолошки различни "
    "пристапи стигнуваат до речиси ист заклучок — силна потврда на наодите од "
    "Фаза 6."
)

h2("7.2 Structural Balance")
body(
    "<b>Идеја:</b> во здрави социјални мрежи, триади од 3 меѓусебно поврзани "
    "адреси тежнеат да бидат „балансирани“ (сите 3 врски позитивни, или точно "
    "2 негативни). Скриптата ги наоѓа сите триади и брои парни (балансирани) "
    "наспроти непарни (небалансирани) негативни врски — независна проверка дали "
    "довербата следи реален социјален образец или е случаен шум."
)

balance_df = pd.read_csv(RESULTS / "balance_triads_summary.csv")
bd = balance_df.copy()
bd["pct_balanced"] = bd["pct_balanced"].round(1)
bd["pct_unbalanced"] = bd["pct_unbalanced"].round(1)
bd = bd[["dataset", "total_triangles", "balanced", "unbalanced", "pct_balanced", "pct_unbalanced"]]
bd.columns = ["Датасет", "Вкупно триади", "Балансирани", "Небалансирани",
              "% Балансирани", "% Небалансирани"]
df_to_table(bd, font_size=8)
story.append(Spacer(1, 10))

body(
    "<b>Наод:</b> <b>87.2%</b> од триадите во Bitcoin OTC и <b>85.6%</b> во "
    "Bitcoin Alpha се балансирани. Ова е конзистентно помеѓу двата независни "
    "датасети, што ни кажува дека довербата тука не е распределена случајно туку "
    "формира препознатлив, повторлив образец — исти заклучок како кај community "
    "detection (Фаза 4) и Fairness/Goodness крос-валидацијата погоре, добиен "
    "преку целосно поинаков метод."
)

# ============================================================ FINAL VERDICT
story.append(PageBreak())
h1("8. Финален вердикт и заклучоци")

story.append(Paragraph("Клучни заклучоци", styles["H2"]))
verdict_points = [
    "<b>Структура:</b> и двете мрежи се small-world, hub-driven графови со еден "
    "доминантен поврзан компонент (>99.8% од јазлите), мал дијаметар (9-10) и "
    "просечен пат од ~3.6 — доверба/трансакции можат да достигнат речиси секој јазол "
    "за само неколку чекори.",

    "<b>Централност:</b> секоја мрежа е доминирана од еден-два „супер-јазли“ "
    "(node 35 во OTC, node 1 во Alpha) кои се централни по секоја метрика и биле "
    "централни уште од раните фази на мрежата — веројатно рано-регистрирани "
    "маркет-мејкери, а не органски-израснати hub-ови.",

    "<b>Заедници:</b> и двете мрежи имаат силна, значајна community-структура "
    "(модуларност 0.47-0.49, 18-21 заедници), без ниту една доминантна заедница — "
    "мрежата е федерација на споредливо-големи довербени кластери, а не еден "
    "монолитен блок.",

    "<b>Временска динамика:</b> активноста не е рамномерна — постојат остри "
    "временски избувнувања (особено јуни 2011, со 6-9x над-просечна активност), "
    "што укажува дека растот на мрежата бил воден од настани (пр. надворешни "
    "медиумски/пазарни случувања), а не постојан органски прилив.",

    "<b>Доверба/аномалии:</b> мрежите се доминантно позитивни (>85% позитивни "
    "оценки), но постои мала, јасно-идентификувана подгрупа адреси (158 во OTC, "
    "51 во Alpha) со систематски негативни оценки од многу различни извори — "
    "веродостоен сигнал за реално distrusted актери. Дополнително, детектирани се "
    "групи на bursty-rating настани со сомнителен, координиран образец на "
    "секвенцијални ID-ови во ист ден, што заслужува посебна проверка за sybil "
    "однесување.",
]
for p in verdict_points:
    story.append(Paragraph("• " + p, styles["Verdict"]))

story.append(Spacer(1, 10))
h2("Ограничувања")
body(
    "Овој извештај се потпира на two SNAP rating-мрежи како proxy за Bitcoin "
    "трансакциско поврзување, не на целосен on-chain граф — заклучоците важат за "
    "„who-trusts-whom“ односи на две конкретни платформи, не за целата Bitcoin "
    "мрежа. Betweenness centrality е апроксимирана (k=500 sampling) поради "
    "пресметковна цена. Аномалиите се детектирани со едноставни, објаснливи "
    "хевристики (threshold-based), не со supervised машинско учење — тие се "
    "кандидати за истрага, не потврдени измами."
)

h2("Општ вердикт")
body(
    "<b>Двете мрежи покажуваат здрава, реалистична структура на социјална/"
    "трансакциска доверба: концентрирана околу неколку рано-воспоставени hub-ови, "
    "организирана во значаен број препознатливи заедници, со активност водена од "
    "надворешни настани, и со мала но реална подгрупа на систематски distrusted "
    "и потенцијално sybil-controlled адреси.</b> Наодите се конзистентни помеѓу "
    "двата независни датасети (OTC и Alpha), што ја зголемува довербата во "
    "робустноста на заклучоците наспроти артефакт на еден единствен датасет."
)

# ============================================================ BUILD
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawRightString(A4[0] - 0.7 * inch, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT_PDF), pagesize=A4,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    topMargin=0.85 * inch, bottomMargin=0.85 * inch,
    title="Bitcoin Transaction Network - Final Report",
)
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Report written to {OUT_PDF}")
