#!/usr/bin/env python3
"""Deterministic corpus builder for fast-transformer distillation.

Writes one JSON object per line to ``notes/distill/corpus.jsonl``:

    {"utt_id", "text", "lang", "reference_id", "temperature", "seed"}

Text is generated programmatically from per-language template banks and phrase
pools with a single seeded RNG, so the same ``--seed``/``--count`` always
produce the same corpus - the capture tool derives its per-utterance torch seed
from ``utt_id``, and reproducing a shard means reproducing the line that made
it. No network, no external corpora.

Style target is companion-bot dialogue (greetings, reactions, opinions, small
stories, questions, quoted speech, plus a formal register), not encyclopedic
prose, because that is what the deployed voice actually says.

    python tools/distill/build_corpus.py
    python tools/distill/build_corpus.py --count 200 --out /tmp/mini.jsonl
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import click

# --------------------------------------------------------------------------
# Global knobs
# --------------------------------------------------------------------------

DEFAULT_SEED = 20260810
DEFAULT_COUNT = 2500
DEFAULT_OUT = Path("notes/distill/corpus.jsonl")

# Share of the corpus per language.
LANG_MIX = {"en": 0.40, "id": 0.40, "ja": 0.20}

# Share of utterances per length class.
LENGTH_MIX = {"short": 0.50, "medium": 0.35, "long": 0.15}

# Share of utterances carrying at least one inline emotion tag.
TAG_RATE = 0.20

# Share of utterances generated at the production temperature.
HOT_TEMPERATURE_RATE = 0.70
HOT_TEMPERATURE = 0.7
COLD_TEMPERATURE = 0.2

# Weighted round robin, 2:1:1. Repeated inline rather than expressed as
# weights so the on-disk order is obvious and stable.
REFERENCE_CYCLE = ["beatrice10", "beatrice30", "beatrice10", "default"]

EMOTION_TAGS = [
    "[laugh]",
    "[laughs]",
    "[whispers]",
    "[sighs]",
    "[sad]",
    "[excited]",
    "[angry]",
    "[chuckles]",
]

# A "short" utterance is one sentence of 5-15 words. Japanese has no spaces, so
# it is measured in characters instead; ~2.2 chars per word puts the equivalent
# band at roughly 12-34.
SHORT_BAND = {"en": (5, 15), "id": (5, 15), "ja": (12, 34)}

# Sentence sequences per length class. Each entry is a list of template
# categories; picking a shape rather than sampling categories independently is
# what keeps multi-sentence utterances reading like one thought.
SHAPES = {
    "short": [
        ["open"],
        ["ask"],
        ["react"],
        ["opinion"],
        ["story"],
        ["quote"],
        ["formal"],
        ["close"],
    ],
    "medium": [
        ["react", "opinion"],
        ["open", "ask"],
        ["opinion", "ask"],
        ["story", "react"],
        ["quote", "react"],
        ["formal", "formal"],
        ["react", "opinion", "ask"],
        ["open", "story", "ask"],
        ["story", "story", "react"],
        ["opinion", "story"],
        ["story", "opinion"],
        ["react", "close"],
        ["formal", "ask"],
        ["open", "opinion"],
        ["quote", "opinion"],
        ["ask", "opinion"],
        ["story", "ask"],
        ["open", "react", "ask"],
        ["opinion", "opinion", "close"],
        ["story", "quote"],
    ],
    "long": [
        ["open", "story", "story", "opinion", "ask"],
        ["react", "story", "story", "story", "close"],
        ["formal", "formal", "opinion", "ask"],
        ["open", "quote", "react", "opinion", "ask", "close"],
        ["story", "story", "quote", "react", "opinion"],
        ["open", "opinion", "story", "story", "close"],
        ["react", "opinion", "story", "ask", "close"],
        ["story", "story", "story", "opinion", "ask", "close"],
        ["formal", "formal", "formal", "opinion"],
        ["open", "story", "quote", "react", "close"],
        ["story", "opinion", "opinion", "story", "ask"],
        ["react", "quote", "story", "opinion", "close"],
    ],
}

# --------------------------------------------------------------------------
# English
# --------------------------------------------------------------------------

EN_POOLS = {
    "noun_thing": [
        "your voice message",
        "the rain last night",
        "that old photo album",
        "this chipped blue mug",
        "the new bakery downtown",
        "your handwriting",
        "that song you sent",
        "the last train home",
        "my neighbor's cat",
        "this ridiculous group chat",
        "the smell of coffee",
        "your terrible pun",
        "that documentary about octopuses",
        "the plant on my windowsill",
        "this second-hand paperback",
        "the streetlight outside",
        "your birthday plan",
        "that half-finished puzzle",
        "the sound of rain",
        "my grandmother's recipe",
        "this tiny notebook",
        "the bus that never comes",
        "your favorite hoodie",
        "that weird dream",
    ],
    "activity": [
        "walking home in the dark",
        "rewatching the same episode",
        "burning toast again",
        "arguing about pineapple pizza",
        "learning three chords badly",
        "reorganizing my bookshelf",
        "napping through the afternoon",
        "waiting for the kettle",
        "singing off-key in the shower",
        "counting rooftops from the balcony",
        "reading past midnight",
        "pretending to be productive",
        "cleaning out my inbox",
        "walking the long way around",
        "talking to my plants",
        "collecting bus tickets",
        "watching the fog roll in",
        "baking something slightly wrong",
        "learning your favorite recipe",
        "arguing with the vending machine",
        "listening to old voicemails",
        "wandering through the night market",
        "folding laundry very slowly",
        "staring at the ceiling",
    ],
    "feeling": [
        "exhausted",
        "weirdly calm",
        "a little homesick",
        "genuinely delighted",
        "half awake",
        "strangely hopeful",
        "quietly proud",
        "completely lost",
        "oddly nostalgic",
        "kind of restless",
        "surprisingly okay",
        "a bit dramatic",
        "stubbornly cheerful",
        "overwhelmed",
        "content",
        "jittery",
        "unbothered",
        "sentimental",
        "wide awake",
        "grumpy",
        "hopeful",
        "fuzzy-headed",
        "warm",
        "antsy",
    ],
    "place": [
        "the riverside park",
        "the noodle place by the office",
        "that rooftop bar",
        "the old library",
        "the corner store",
        "the night market",
        "the station platform",
        "my tiny kitchen",
        "the back of the bus",
        "the empty parking lot",
        "the beach in winter",
        "the stairwell",
        "that quiet bookshop",
        "the laundromat",
        "the hill behind school",
        "the diner that never closes",
        "the greenhouse",
        "the pier",
        "my favorite bench",
        "the alley with the murals",
        "the rooftop",
        "the train window",
    ],
    "food": [
        "cold soba",
        "instant coffee",
        "too much garlic bread",
        "a suspicious sandwich",
        "late-night dumplings",
        "burnt cookies",
        "mango sticky rice",
        "leftover curry",
        "peppermint tea",
        "a whole watermelon",
        "street corn",
        "overpriced pastries",
        "fried rice at midnight",
        "salted caramel anything",
        "my failed omelette",
        "spicy noodles",
        "iced coffee at eleven at night",
        "grandma's soup",
        "questionable convenience store sushi",
        "fresh bread",
        "cheap chocolate",
        "tomato soup",
    ],
    "quote": [
        "I'll be there in five minutes",
        "you always say that",
        "this is the last one, I promise",
        "it's not that deep",
        "trust me on this",
        "I told you so",
        "we're never doing that again",
        "that's exactly what I needed",
        "don't wait up for me",
        "I think I'm lost",
        "you'd have loved it",
        "call me when you land",
        "I'm not crying, you are",
        "let's take the long way",
        "I forgot how loud it gets",
        "it looked better online",
        "I'll handle it",
        "some things you can't rush",
        "I saved you a seat",
        "just five more minutes",
        "that was not the plan",
        "we should do this more often",
        "I'm proud of you",
        "it's going to be fine",
    ],
    "person": [
        "my old roommate",
        "the barista with the tattoos",
        "my little brother",
        "the guy at the bus stop",
        "my aunt",
        "the security guard",
        "my coworker Ren",
        "the woman upstairs",
        "my best friend",
        "the delivery driver",
        "my landlord",
        "the piano teacher",
        "my cousin",
        "the shopkeeper",
        "a stranger on the train",
        "the night nurse",
        "my dad",
        "the kid next door",
        "my study partner",
        "the taxi driver",
        "my old teacher",
        "the librarian",
    ],
    "adjpos": [
        "lovely",
        "ridiculous in the best way",
        "perfect",
        "so warm",
        "kind of magical",
        "genuinely funny",
        "gorgeous",
        "comforting",
        "clever",
        "sweet",
        "unreal",
        "the best thing all week",
        "really something",
        "charming",
        "brilliant",
        "cozy",
        "surprisingly good",
        "wonderful",
        "adorable",
        "impressive",
        "delightful",
        "exactly right",
    ],
    "adjneg": [
        "a disaster",
        "a bit much",
        "exhausting",
        "completely unnecessary",
        "so awkward",
        "a mess",
        "overrated",
        "frustrating",
        "the worst timing",
        "kind of sad",
        "confusing",
        "too loud",
        "unfair",
        "a little scary",
        "cursed",
        "annoying",
        "hopeless",
        "cold and grey",
        "a waste",
        "embarrassing",
    ],
    "timeword": [
        "this morning",
        "last night",
        "around three in the morning",
        "on the way home",
        "earlier today",
        "last Tuesday",
        "all weekend",
        "before the rain started",
        "just now",
        "the other day",
        "during the storm",
        "right after lunch",
        "late last night",
        "first thing tomorrow",
        "in the middle of class",
        "on my day off",
        "after the concert",
        "this whole week",
        "yesterday evening",
        "sometime in April",
    ],
    "plan": [
        "go for a long walk",
        "finally clean the kitchen",
        "try that ramen place",
        "learn one new song",
        "call my mom",
        "watch the sunrise",
        "fix the wobbly chair",
        "write actual letters",
        "take the slow train",
        "bake something ambitious",
        "reread that book",
        "visit the aquarium",
        "sort out my photos",
        "plant something on the balcony",
        "go swimming at dawn",
        "make a proper breakfast",
        "tidy the closet",
        "start that puzzle",
        "walk to the top of the hill",
        "cook for everyone",
        "see the meteor shower",
        "learn to whistle properly",
    ],
    "music": [
        "that slow piano track",
        "the song you hummed",
        "an old jazz record",
        "the playlist you made",
        "a very dramatic string section",
        "that one chorus",
        "a lo-fi loop",
        "the soundtrack from that film",
        "a busker's cover",
        "the album on repeat",
        "some ridiculous pop song",
        "a lullaby my mom sang",
        "the bass line",
        "the acoustic version",
        "a live recording with the crowd singing",
        "that bittersweet closing track",
        "the intro riff",
        "an accordion in the street",
        "the demo version",
        "a song in a language I don't speak",
    ],
    "weather": [
        "the fog",
        "the first real cold of the year",
        "that sudden downpour",
        "a warm wind",
        "the grey sky",
        "thunder in the distance",
        "the drizzle",
        "unreasonable heat",
        "the frost on the window",
        "clear skies for once",
        "the storm rolling in",
        "a soft rain",
        "the humidity",
        "snow that didn't stick",
        "the last warm evening",
        "a sky full of stars",
        "the wind off the water",
        "that sticky afternoon heat",
        "an early sunset",
        "the smell before rain",
    ],
    "small_thing": [
        "a chipped mug",
        "a folded receipt",
        "a stray sticker",
        "one mismatched sock",
        "a pressed flower",
        "a train ticket stub",
        "a bent paperclip",
        "a coin from another country",
        "a half-melted candle",
        "a postcard nobody sent",
        "a shoelace with a knot",
        "a button from a coat",
        "a dried leaf in a book",
        "a keychain missing its key",
        "a tiny paper crane",
        "an old bus pass",
        "a smudged polaroid",
        "a rubber band ball",
        "a cracked phone case",
        "a note in the margin",
    ],
    "abstract": [
        "the way people say goodbye",
        "how quickly a week disappears",
        "small kindnesses",
        "the pause before a song ends",
        "the courage it takes to start over",
        "how memory edits itself",
        "the comfort of routine",
        "the difference a day makes",
        "how strangers become familiar",
        "the weight of an unsent message",
        "the quiet after a party",
        "how much a name can carry",
        "the honesty of tired people",
        "the pull of an old street",
        "how endings sneak up",
        "the value of doing nothing",
        "the shape of a good silence",
        "how a place remembers you",
        "the way we forgive slowly",
        "the small bravery of asking",
    ],
}

EN_TEMPLATES = {
    "open": [
        "Hey, you're back. I was just thinking about {noun_thing}.",
        "Good morning! Did you sleep at all, or were you {activity} again?",
        "Oh, hi. I've had {noun_thing} stuck in my head all day.",
        "Welcome back. It got very quiet without you.",
        "Hey there. I've been {activity} while you were gone.",
        "There you are! I saved you a story about {person}.",
        "Hi. Honestly, I'm feeling {feeling} today.",
        "Evening. {weather} finally showed up, by the way.",
        "Hey, one quick question before I forget about {noun_thing}.",
        "You made it. I've been waiting since {timeword}.",
    ],
    "ask": [
        "So what did you think of {noun_thing}?",
        "Do you ever catch yourself {activity}?",
        "Should we {plan} this weekend?",
        "How do you feel about {food}, honestly?",
        "Have you been to {place} yet?",
        "What would you say to {person} if you had the chance?",
        "Why does {abstract} bother me so much?",
        "Can we talk about {music} for a second?",
        "Are you free {timeword}, or is that a bad idea?",
        "What's the last thing that made you feel {feeling}?",
    ],
    "react": [
        "Okay, that's {adjpos} and I refuse to hear otherwise.",
        "Oh no. That sounds {adjneg}.",
        "Wait, really? That's {adjpos}.",
        "See, this is exactly why I like talking to you.",
        "That's the funniest thing I've heard {timeword}.",
        "I can't believe you did that without telling me.",
        "Honestly? {adjpos}. Full stop.",
        "Ugh, {adjneg}. I'm sorry it went that way.",
        "You're kidding. {person} actually said that?",
        "That got me right in the chest, not going to lie.",
    ],
    "opinion": [
        "I think {abstract} is underrated.",
        "For what it's worth, {noun_thing} still gets me every time.",
        "I would trade almost anything for {food} right now.",
        "{music} does something to me I can't explain.",
        "I've decided {activity} counts as self-care.",
        "Nothing beats {place} when it's empty.",
        "I keep {small_thing} because throwing it away feels wrong.",
        "People underestimate how much {weather} changes a day.",
        "If you ask me, we should all {plan} more often.",
        "I don't think {abstract} gets talked about enough.",
    ],
    "story": [
        "{timeword} I was {activity} and completely lost track of time.",
        "I ran into {person} at {place}, of all places.",
        "There was {weather}, and the whole street went quiet.",
        "I found {small_thing} in an old coat pocket.",
        "We ended up at {place} eating {food} at some absurd hour.",
        "{person} tried to teach me a card game and gave up halfway.",
        "The power went out, so we just sat there listening to {music}.",
        "I spent the whole afternoon {activity}, and I regret nothing.",
        "Someone left {small_thing} on my doorstep with no note.",
        "It started raining right as we reached {place}.",
    ],
    "quote": [
        'She looked at me and said, "{quote}"',
        'And then {person} goes, "{quote}" - just like that.',
        'I remember my mother saying, "{quote}" whenever it rained.',
        '"{quote}" - that\'s what he texted, nothing else.',
        'The note just read, "{quote}"',
        '{person} shrugged and said, "{quote}"',
        'I keep hearing that one line: "{quote}"',
        'You once told me, "{quote}" and I never forgot it.',
        'The last thing she said was, "{quote}"',
        'He laughed and said, "{quote}" like it was obvious.',
    ],
    "formal": [
        "Good evening. I trust the day treated you kindly.",
        "Thank you for your patience; the matter has been resolved.",
        "I would like to note, respectfully, that {abstract} deserves more attention.",
        "Please accept my apologies for the delay in responding.",
        "If it is convenient, we could discuss {noun_thing} tomorrow.",
        "It is a pleasure to speak with you again.",
        "I have prepared a brief summary regarding {noun_thing}.",
        "Allow me to say that {adjpos} hardly covers it.",
        "Kindly let me know whether {timeword} would suit your schedule.",
        "On reflection, I believe we should {plan} before the season ends.",
    ],
    "close": [
        "Anyway, get some rest, okay?",
        "Alright, I'll stop talking now. Mostly.",
        "Tell me how it goes, seriously.",
        "I'll be right here when you get back.",
        "Okay, go do the thing. You've got this.",
        "Sleep well, and don't overthink it.",
        "Talk soon. Drink some water.",
        "That's all from me. Go be brilliant.",
        "Message me when you're home safe.",
        "Okay, enough from me. Your turn.",
    ],
}

# --------------------------------------------------------------------------
# Indonesian (colloquial aku/kamu, with a formal saya/Anda register)
# --------------------------------------------------------------------------

ID_POOLS = {
    "noun_thing": [
        "pesan suara kamu",
        "hujan semalam",
        "album foto lama itu",
        "gelas biru yang retak",
        "toko roti baru di ujung jalan",
        "tulisan tangan kamu",
        "lagu yang kamu kirim",
        "kereta terakhir malam ini",
        "kucing tetangga",
        "grup chat yang absurd itu",
        "aroma kopi pagi",
        "lelucon garing kamu",
        "dokumenter tentang gurita itu",
        "tanaman di jendela kamarku",
        "buku bekas yang aku beli",
        "lampu jalan di depan rumah",
        "rencana ulang tahun kamu",
        "teka-teki yang belum selesai",
        "suara hujan di atap",
        "resep nenekku",
        "buku catatan kecilku",
        "angkot yang tak kunjung lewat",
        "hoodie kesayangan kamu",
        "mimpi aneh semalam",
    ],
    "activity": [
        "jalan kaki pulang malam-malam",
        "nonton ulang episode yang sama",
        "gosongin roti lagi",
        "debat soal nanas di pizza",
        "belajar tiga kunci gitar",
        "beresin rak buku",
        "tidur siang kelamaan",
        "nungguin air mendidih",
        "nyanyi fals di kamar mandi",
        "ngitung atap dari balkon",
        "baca sampai lewat tengah malam",
        "pura-pura sibuk",
        "bersihin inbox",
        "ambil jalan memutar",
        "ngobrol sama tanaman",
        "ngumpulin struk belanja",
        "liatin kabut turun",
        "bikin kue yang agak gagal",
        "belajar masak resep kamu",
        "berantem sama mesin minuman",
        "dengerin voice note lama",
        "keliling pasar malam",
        "lipat baju pelan-pelan",
        "bengong liatin langit-langit",
    ],
    "feeling": [
        "capek banget",
        "anteng aneh",
        "kangen rumah",
        "seneng banget",
        "setengah sadar",
        "optimis tiba-tiba",
        "bangga diam-diam",
        "bingung total",
        "nostalgia",
        "gelisah dikit",
        "ternyata baik-baik aja",
        "lebay dikit",
        "tetep ceria",
        "kewalahan",
        "tenang",
        "deg-degan",
        "santai aja",
        "melow",
        "melek banget",
        "bete",
        "penuh harap",
        "pusing ringan",
        "hangat",
        "gak bisa diem",
    ],
    "place": [
        "taman di pinggir kali",
        "warung mie dekat kantor",
        "kafe di rooftop",
        "perpustakaan lama",
        "warung sebelah",
        "pasar malam",
        "peron stasiun",
        "dapur kecilku",
        "bangku belakang bus",
        "parkiran kosong",
        "pantai pas musim hujan",
        "tangga darurat",
        "toko buku yang sepi",
        "tempat laundry",
        "bukit di belakang sekolah",
        "warteg dua puluh empat jam",
        "rumah kaca",
        "dermaga",
        "bangku favoritku",
        "gang yang penuh mural",
        "atap kosan",
        "jendela kereta",
    ],
    "food": [
        "soto panas",
        "kopi sachet",
        "roti bawang kebanyakan",
        "roti isi yang mencurigakan",
        "dimsum tengah malam",
        "kue gosong",
        "es teh manis",
        "kari sisa kemarin",
        "teh melati",
        "semangka satu buah utuh",
        "jagung bakar",
        "kue mahal di mall",
        "nasi goreng jam dua pagi",
        "apa pun yang rasa karamel",
        "telur dadarku yang gagal",
        "mie pedas level lima",
        "kopi susu jam sebelas malam",
        "sup buatan nenek",
        "sushi minimarket",
        "roti baru keluar oven",
        "cokelat murah",
        "sup tomat",
    ],
    "quote": [
        "aku ke sana lima menit lagi",
        "kamu selalu bilang gitu",
        "ini yang terakhir, janji",
        "gak usah dibawa serius",
        "percaya deh sama aku",
        "tuh kan, udah kubilang",
        "kita gak akan ngulang itu lagi",
        "itu persis yang aku butuhin",
        "gak usah ditungguin",
        "kayaknya aku nyasar",
        "kamu pasti suka",
        "kabarin kalau udah sampai",
        "aku gak nangis kok",
        "kita lewat jalan yang jauh aja",
        "aku lupa serame ini",
        "aslinya beda sama fotonya",
        "biar aku yang urus",
        "ada hal yang gak bisa diburu-buru",
        "aku simpenin tempat buat kamu",
        "lima menit lagi ya",
        "ini di luar rencana",
        "kita harus lebih sering gini",
        "aku bangga sama kamu",
        "semuanya bakal baik-baik aja",
    ],
    "person": [
        "teman kosku dulu",
        "barista yang bertato",
        "adikku",
        "bapak-bapak di halte",
        "tanteku",
        "satpam komplek",
        "rekan kerjaku, Ren",
        "ibu yang di lantai atas",
        "sahabatku",
        "kurir paket",
        "ibu kosku",
        "guru pianoku",
        "sepupuku",
        "penjaga toko",
        "orang asing di kereta",
        "perawat jaga malam",
        "ayahku",
        "anak tetangga",
        "teman belajarku",
        "sopir taksi",
        "guru SD-ku",
        "penjaga perpustakaan",
    ],
    "adjpos": [
        "manis banget",
        "absurd tapi bagus",
        "sempurna",
        "hangat banget",
        "agak ajaib",
        "lucu beneran",
        "cantik",
        "bikin tenang",
        "pinter",
        "baik banget",
        "gak masuk akal saking bagusnya",
        "hal terbaik minggu ini",
        "luar biasa",
        "menawan",
        "keren",
        "nyaman",
        "ternyata enak",
        "indah",
        "gemesin",
        "mengesankan",
        "menyenangkan",
        "pas banget",
    ],
    "adjneg": [
        "kacau",
        "agak berlebihan",
        "melelahkan",
        "gak perlu banget",
        "canggung banget",
        "berantakan",
        "kelewat dipuji",
        "bikin frustrasi",
        "timingnya jelek",
        "agak sedih",
        "membingungkan",
        "berisik banget",
        "gak adil",
        "agak serem",
        "sial banget",
        "nyebelin",
        "gak ada harapan",
        "dingin dan kelabu",
        "sia-sia",
        "memalukan",
    ],
    "timeword": [
        "tadi pagi",
        "semalam",
        "sekitar jam tiga pagi",
        "pas perjalanan pulang",
        "tadi siang",
        "Selasa kemarin",
        "sepanjang akhir pekan",
        "sebelum hujan turun",
        "barusan",
        "waktu itu",
        "pas badai",
        "habis makan siang",
        "larut malam kemarin",
        "besok pagi-pagi",
        "pas lagi kelas",
        "pas hari libur",
        "habis konser",
        "seminggu ini",
        "kemarin sore",
        "sekitar bulan April",
    ],
    "plan": [
        "jalan-jalan jauh",
        "beresin dapur",
        "cobain tempat ramen itu",
        "belajar satu lagu baru",
        "telepon ibuku",
        "liat matahari terbit",
        "benerin kursi yang goyang",
        "nulis surat beneran",
        "naik kereta ekonomi",
        "bikin kue yang ribet",
        "baca ulang buku itu",
        "ke akuarium",
        "rapiin foto-foto lama",
        "nanam sesuatu di balkon",
        "berenang pagi-pagi",
        "masak sarapan lengkap",
        "beresin lemari",
        "mulai teka-teki itu",
        "naik sampai puncak bukit",
        "masak buat semua orang",
        "liat hujan meteor",
        "belajar siul yang bener",
    ],
    "music": [
        "lagu piano yang pelan itu",
        "lagu yang kamu senandungin",
        "piringan hitam jazz lama",
        "playlist buatan kamu",
        "bagian biola yang dramatis",
        "reff-nya yang itu",
        "lagu lo-fi",
        "soundtrack film itu",
        "cover pengamen jalanan",
        "album yang aku puter terus",
        "lagu pop yang norak",
        "lagu nina bobo dari ibuku",
        "bass line-nya",
        "versi akustiknya",
        "rekaman live yang penontonnya ikut nyanyi",
        "lagu penutup yang getir",
        "riff pembukanya",
        "suara akordeon di jalan",
        "versi demo-nya",
        "lagu bahasa asing yang aku gak ngerti",
    ],
    "weather": [
        "kabut tipis",
        "dingin pertama tahun ini",
        "hujan deras tiba-tiba",
        "angin hangat",
        "langit kelabu",
        "suara guntur di kejauhan",
        "gerimis",
        "panas yang gak masuk akal",
        "embun di kaca jendela",
        "langit cerah buat sekali ini",
        "badai yang mendekat",
        "hujan lembut",
        "udara lembap",
        "salju yang gak nempel",
        "sore hangat terakhir",
        "langit penuh bintang",
        "angin dari arah laut",
        "panas siang yang lengket",
        "matahari terbenam lebih awal",
        "bau tanah sebelum hujan",
    ],
    "small_thing": [
        "gelas yang gompal",
        "struk yang terlipat",
        "stiker nyasar",
        "kaus kaki yang gak sepasang",
        "bunga kering di buku",
        "potongan tiket kereta",
        "klip kertas bengkok",
        "koin dari negara lain",
        "lilin setengah meleleh",
        "kartu pos yang gak pernah dikirim",
        "tali sepatu yang simpul",
        "kancing jaket",
        "daun kering di halaman buku",
        "gantungan kunci tanpa kunci",
        "burung kertas kecil",
        "kartu bus lama",
        "polaroid yang buram",
        "bola karet gelang",
        "case HP yang retak",
        "catatan di pinggir halaman",
    ],
    "abstract": [
        "cara orang pamitan",
        "betapa cepat seminggu lewat",
        "kebaikan-kebaikan kecil",
        "jeda sebelum lagu selesai",
        "keberanian buat mulai lagi",
        "cara ingatan menyunting dirinya",
        "kenyamanan rutinitas",
        "beda satu hari yang ternyata besar",
        "cara orang asing jadi akrab",
        "berat dari pesan yang gak jadi dikirim",
        "sepi setelah pesta",
        "seberapa banyak arti satu nama",
        "kejujuran orang yang kecapekan",
        "tarikan dari jalan lama",
        "cara akhir datang diam-diam",
        "nilai dari nggak ngapa-ngapain",
        "bentuk dari diam yang enak",
        "cara sebuah tempat mengingat kita",
        "cara kita memaafkan pelan-pelan",
        "keberanian kecil buat bertanya",
    ],
}

ID_TEMPLATES = {
    "open": [
        "Hei, kamu balik juga. Aku barusan mikirin {noun_thing}.",
        "Pagi! Kamu tidur nggak sih, atau {activity} lagi semalaman?",
        "Eh, hai. {noun_thing} nempel terus di kepalaku seharian.",
        "Selamat datang kembali. Sepi banget tadi tanpa kamu.",
        "Hai. Tadi aku {activity} sambil nungguin kamu.",
        "Nah, itu dia! Aku nyimpen cerita soal {person} buat kamu.",
        "Hai. Jujur aja, hari ini aku {feeling}.",
        "Malam. Ngomong-ngomong, {weather} akhirnya datang juga.",
        "Hei, satu pertanyaan dulu sebelum aku lupa soal {noun_thing}.",
        "Akhirnya sampai juga. Aku nunggu dari {timeword}.",
    ],
    "ask": [
        "Jadi menurut kamu {noun_thing} gimana?",
        "Kamu pernah nggak tiba-tiba {activity}?",
        "Kita {plan} akhir pekan ini, yuk?",
        "Jujur deh, kamu suka {food} nggak?",
        "Kamu udah pernah ke {place} belum?",
        "Kalau ketemu {person} lagi, kamu mau bilang apa?",
        "Kenapa ya {abstract} bikin aku kepikiran terus?",
        "Boleh nggak kita ngobrolin {music} sebentar?",
        "Kamu kosong {timeword}, atau itu ide yang buruk?",
        "Terakhir kali kamu ngerasa {feeling} itu kapan?",
    ],
    "react": [
        "Oke, itu {adjpos} dan aku nggak nerima bantahan.",
        "Yah, kedengarannya {adjneg} banget.",
        "Serius? Itu {adjpos}.",
        "Nah, ini alasan aku suka ngobrol sama kamu.",
        "Itu hal paling lucu yang aku denger {timeword}.",
        "Nggak nyangka kamu ngelakuin itu tanpa bilang aku.",
        "Jujur? {adjpos}. Titik.",
        "Aduh, {adjneg}. Maaf ya jadinya begitu.",
        "Bohong. {person} beneran ngomong gitu?",
        "Itu ngena banget di dada, nggak bohong.",
    ],
    "opinion": [
        "Menurutku {abstract} itu kurang dihargai.",
        "Buat aku sih, {noun_thing} selalu bikin baper.",
        "Aku rela tukar apa aja demi {food} sekarang.",
        "{music} bikin aku ngerasa sesuatu yang susah dijelasin.",
        "Aku memutuskan kalau {activity} itu termasuk self-care.",
        "Nggak ada yang ngalahin {place} pas lagi sepi.",
        "Aku nyimpen {small_thing} karena buang rasanya salah.",
        "Orang suka nggak sadar {weather} bisa ngubah satu hari.",
        "Kalau nanya aku, kita semua harus lebih sering {plan}.",
        "Kayaknya {abstract} jarang banget dibahas orang.",
    ],
    "story": [
        "{timeword} aku {activity} sampai lupa waktu.",
        "Aku ketemu {person} di {place}, dari sekian banyak tempat.",
        "Tadi ada {weather}, terus seluruh jalanan mendadak sunyi.",
        "Aku nemu {small_thing} di saku jaket lama.",
        "Kita berakhir di {place} sambil makan {food} jam segitu.",
        "{person} nyoba ngajarin aku main kartu terus nyerah di tengah.",
        "Listriknya mati, jadi kita cuma dengerin {music}.",
        "Aku habisin sore buat {activity}, dan nggak nyesel sama sekali.",
        "Ada yang naruh {small_thing} di depan pintu tanpa pesan.",
        "Hujan mulai turun pas kita sampai {place}.",
    ],
    "quote": [
        'Dia natap aku terus bilang, "{quote}"',
        'Terus {person} nyeletuk, "{quote}" - gitu aja.',
        'Aku inget ibuku selalu bilang, "{quote}" tiap hujan.',
        '"{quote}" - cuma itu isi pesannya.',
        'Di catatannya cuma tertulis, "{quote}"',
        '{person} cuma angkat bahu terus bilang, "{quote}"',
        'Kalimat itu kebayang terus: "{quote}"',
        'Kamu pernah bilang, "{quote}" dan aku nggak pernah lupa.',
        'Kalimat terakhirnya, "{quote}"',
        'Dia ketawa terus bilang, "{quote}" kayak itu hal biasa.',
    ],
    "formal": [
        "Selamat malam. Semoga hari Anda berjalan dengan baik.",
        "Terima kasih atas kesabarannya; persoalannya sudah kami selesaikan.",
        "Dengan hormat, saya rasa {abstract} layak mendapat perhatian lebih.",
        "Mohon maaf atas keterlambatan tanggapan kami.",
        "Jika berkenan, kita bisa membahas {noun_thing} besok.",
        "Suatu kehormatan dapat berbicara dengan Anda lagi.",
        "Saya sudah menyiapkan ringkasan singkat mengenai {noun_thing}.",
        "Izinkan saya mengatakan bahwa {adjpos} pun rasanya belum cukup.",
        "Mohon kabari saya apakah {timeword} sesuai dengan jadwal Anda.",
        "Setelah dipikir ulang, sebaiknya kita {plan} sebelum musim berganti.",
    ],
    "close": [
        "Ya udah, istirahat dulu ya.",
        "Oke, aku berhenti ngomong sekarang. Kurang lebih.",
        "Nanti ceritain gimana hasilnya, serius.",
        "Aku di sini kok kalau kamu balik.",
        "Oke, sana kerjain. Kamu pasti bisa.",
        "Tidur yang nyenyak, jangan kebanyakan mikir.",
        "Ngobrol lagi nanti. Minum air dulu.",
        "Segitu dulu dari aku. Sana, jadi hebat.",
        "Kabarin ya kalau udah sampai rumah.",
        "Oke, cukup aku yang ngoceh. Giliran kamu.",
    ],
}

# --------------------------------------------------------------------------
# Japanese (casual + polite; verbs stay in dictionary form so templates can
# attach their own endings)
# --------------------------------------------------------------------------

JA_POOLS = {
    "noun_thing": [
        "きみのボイスメッセージ",
        "昨日の夜の雨",
        "古いアルバム",
        "縁の欠けた青いマグカップ",
        "駅前にできた新しいパン屋",
        "きみの字",
        "きみが送ってくれた曲",
        "終電",
        "となりの家の猫",
        "あのくだらないグループチャット",
        "朝のコーヒーの匂い",
        "きみのしょうもない冗談",
        "タコのドキュメンタリー",
        "窓辺の観葉植物",
        "古本屋で買った文庫本",
        "家の前の街灯",
        "きみの誕生日の計画",
        "途中でやめたパズル",
        "屋根に落ちる雨の音",
        "祖母のレシピ",
        "小さなノート",
        "なかなか来ないバス",
        "きみのお気に入りのパーカー",
        "変な夢",
    ],
    "activity": [
        "夜道を歩いて帰る",
        "同じ話を何度も見返す",
        "またトーストを焦がす",
        "ピザのパイナップル論争をする",
        "ギターのコードを三つ覚える",
        "本棚を並べ直す",
        "昼寝をしすぎる",
        "お湯が沸くのを待つ",
        "お風呂で音痴に歌う",
        "ベランダから屋根を数える",
        "夜更かしして本を読む",
        "忙しいふりをする",
        "受信箱を片づける",
        "わざと遠回りする",
        "植物に話しかける",
        "レシートを集める",
        "霧が降りてくるのを眺める",
        "少し失敗したお菓子を焼く",
        "きみの得意料理を覚える",
        "自販機と格闘する",
        "古いボイスメモを聞く",
        "夜市をぶらぶらする",
        "洗濯物をゆっくり畳む",
        "天井をぼんやり見上げる",
    ],
    "feeling": [
        "くたくた",
        "なぜか穏やか",
        "ちょっとホームシック",
        "すごく幸せ",
        "寝ぼけ気味",
        "急に前向き",
        "こっそり誇らしい気分",
        "迷子気分",
        "なつかしい気分",
        "そわそわ",
        "意外と平気",
        "ちょっと大げさ",
        "意地でも上機嫌",
        "いっぱいいっぱい",
        "満ち足りた気分",
        "どきどき",
        "気楽",
        "しんみり",
        "目が冴えた感じ",
        "不機嫌",
        "希望でいっぱい",
        "頭がぼんやり気味",
        "あたたかい気持ち",
        "落ち着かない感じ",
    ],
    "place": [
        "川沿いの公園",
        "会社の近くのそば屋",
        "屋上のバー",
        "古い図書館",
        "角のコンビニ",
        "夜市",
        "駅のホーム",
        "うちの狭い台所",
        "バスのいちばん後ろの席",
        "からっぽの駐車場",
        "冬の海辺",
        "非常階段",
        "静かな本屋",
        "コインランドリー",
        "学校の裏の坂",
        "二十四時間の定食屋",
        "温室",
        "桟橋",
        "お気に入りのベンチ",
        "壁画のある路地",
        "屋上",
        "電車の窓際",
    ],
    "food": [
        "冷たいそば",
        "インスタントコーヒー",
        "ガーリックトーストの食べすぎ",
        "あやしいサンドイッチ",
        "夜中の餃子",
        "焦げたクッキー",
        "マンゴーのもち米",
        "昨日の残りのカレー",
        "ミントティー",
        "まるごと一個のスイカ",
        "焼きとうもろこし",
        "高すぎるケーキ",
        "夜中のチャーハン",
        "塩キャラメル味のもの",
        "失敗したオムレツ",
        "辛いラーメン",
        "夜十一時のアイスコーヒー",
        "祖母のスープ",
        "コンビニの寿司",
        "焼きたてのパン",
        "安いチョコレート",
        "トマトスープ",
    ],
    "quote": [
        "五分で行くから",
        "いつもそう言うよね",
        "これで最後、約束する",
        "そんなに深い話じゃないよ",
        "わたしを信じて",
        "だから言ったのに",
        "もう二度とやらないからね",
        "それがちょうど欲しかったの",
        "先に寝ててね",
        "たぶん道に迷った",
        "きみなら絶対好きだよ",
        "着いたら連絡してね",
        "泣いてないってば",
        "遠回りして帰ろう",
        "こんなにうるさかったっけ",
        "写真のほうがよかったな",
        "わたしがやっておくよ",
        "急いでどうにかなる話じゃない",
        "席、取っておいたよ",
        "あと五分だけ",
        "こんな予定じゃなかったのに",
        "もっと会おうよ",
        "よくがんばったね",
        "きっと大丈夫だよ",
    ],
    "person": [
        "昔のルームメイト",
        "タトゥーのある店員さん",
        "弟",
        "バス停のおじさん",
        "おば",
        "警備員さん",
        "同僚のレンさん",
        "上の階の人",
        "親友",
        "配達の人",
        "大家さん",
        "ピアノの先生",
        "いとこ",
        "商店街のおばちゃん",
        "電車で隣になった人",
        "夜勤の看護師さん",
        "父",
        "となりの家の子",
        "勉強仲間",
        "タクシーの運転手さん",
        "小学校の先生",
        "図書館の司書さん",
    ],
    "adjpos": [
        "すてき",
        "いい意味でめちゃくちゃ",
        "完璧",
        "すごくあたたかい",
        "ちょっと魔法みたい",
        "本当におもしろい",
        "きれい",
        "ほっとする",
        "賢い",
        "やさしい",
        "現実じゃないみたい",
        "今週いちばんの出来事",
        "なかなかのもの",
        "かわいらしい",
        "見事",
        "居心地がいい",
        "意外とおいしい",
        "すばらしい",
        "愛おしい",
        "立派",
        "うれしい",
        "ちょうどいい",
    ],
    "adjneg": [
        "散々",
        "ちょっとやりすぎ",
        "へとへとになる",
        "まったく必要ない",
        "気まずすぎる",
        "ぐちゃぐちゃ",
        "評価されすぎ",
        "もどかしい",
        "タイミングが最悪",
        "少し切ない",
        "ややこしい",
        "うるさすぎる",
        "不公平",
        "ちょっと怖い",
        "ついてない",
        "うっとうしい",
        "お手上げ",
        "寒くて灰色",
        "むだ",
        "恥ずかしい",
    ],
    "timeword": [
        "今朝",
        "昨日の夜",
        "朝の三時ごろ",
        "帰り道で",
        "今日の昼間",
        "先週の火曜日",
        "週末ずっと",
        "雨が降り出す前に",
        "ついさっき",
        "この前",
        "嵐の最中に",
        "お昼を食べたあと",
        "夜遅くに",
        "明日の朝いちばんに",
        "授業の途中で",
        "休みの日に",
        "ライブのあと",
        "今週ずっと",
        "昨日の夕方",
        "四月ごろ",
    ],
    "plan": [
        "長い散歩に行く",
        "台所をちゃんと片づける",
        "あのラーメン屋を試す",
        "新しい曲を一つ覚える",
        "母に電話する",
        "朝日を見に行く",
        "がたつく椅子を直す",
        "手紙を書く",
        "各駅停車に乗る",
        "手の込んだお菓子を焼く",
        "あの本を読み返す",
        "水族館に行く",
        "昔の写真を整理する",
        "ベランダに何か植える",
        "朝から泳ぎに行く",
        "ちゃんとした朝ごはんを作る",
        "クローゼットを片づける",
        "あのパズルを始める",
        "丘のてっぺんまで歩く",
        "みんなのご飯を作る",
        "流星群を見る",
        "口笛をちゃんと覚える",
    ],
    "music": [
        "あのゆっくりしたピアノの曲",
        "きみが口ずさんでた歌",
        "古いジャズのレコード",
        "きみが作ったプレイリスト",
        "やたら大げさな弦の音",
        "あのサビ",
        "ローファイのループ",
        "あの映画のサントラ",
        "路上ライブのカバー",
        "ずっと繰り返してるアルバム",
        "ばかみたいなポップス",
        "母が歌ってくれた子守唄",
        "ベースライン",
        "アコースティック版",
        "客席も歌ってるライブ音源",
        "切ない終わりの曲",
        "イントロのリフ",
        "街角のアコーディオン",
        "デモ版",
        "意味のわからない外国語の歌",
    ],
    "weather": [
        "霧",
        "今年いちばんの寒さ",
        "急などしゃ降り",
        "あたたかい風",
        "灰色の空",
        "遠くの雷",
        "小雨",
        "ありえない暑さ",
        "窓の霜",
        "めずらしい快晴",
        "近づいてくる嵐",
        "やさしい雨",
        "湿気",
        "積もらない雪",
        "最後のあたたかい夜",
        "星だらけの空",
        "海からの風",
        "べたつく昼の暑さ",
        "早い日暮れ",
        "雨の前のにおい",
    ],
    "small_thing": [
        "縁の欠けたカップ",
        "折りたたんだレシート",
        "はぐれたシール",
        "片方だけの靴下",
        "押し花",
        "電車の切符の半券",
        "曲がったクリップ",
        "外国のコイン",
        "半分溶けたろうそく",
        "出さなかった絵はがき",
        "結び目のある靴ひも",
        "コートのボタン",
        "本にはさんだ枯れ葉",
        "鍵のないキーホルダー",
        "小さな折り鶴",
        "古いバスの定期",
        "にじんだポラロイド",
        "輪ゴムのかたまり",
        "ひびの入ったスマホケース",
        "余白の書き込み",
    ],
    "abstract": [
        "人の別れ方",
        "一週間が消えていく速さ",
        "ささやかなやさしさ",
        "曲が終わる前の一瞬",
        "やり直す勇気",
        "記憶が勝手に書き換わること",
        "くり返しの安心感",
        "一日で変わってしまうこと",
        "他人がなじんでいく過程",
        "送れなかったメッセージの重さ",
        "パーティーのあとの静けさ",
        "名前が背負えるものの多さ",
        "疲れた人の正直さ",
        "昔の通りに引き寄せられる感じ",
        "終わりが静かに近づくこと",
        "何もしない時間の価値",
        "心地よい沈黙のかたち",
        "場所が人を覚えていること",
        "少しずつ許していくこと",
        "たずねるという小さな勇気",
    ],
}

JA_TEMPLATES = {
    "open": [
        "おかえり。ちょうど{noun_thing}のこと考えてたところ。",
        "おはよう!ちゃんと寝た?それともまた{activity}のに夢中だった?",
        "あ、こんにちは。今日はずっと{noun_thing}が頭から離れなくて。",
        "おかえりなさい。いないと静かでしたよ。",
        "やあ。待ってる間、{activity}のに時間を使ってたよ。",
        "来た来た!{person}の話、取っておいたんだ。",
        "こんにちは。正直に言うと、今日は{feeling}です。",
        "こんばんは。そういえば{weather}、やっと来ましたね。",
        "ねえ、忘れないうちに{noun_thing}のこと聞いてもいい?",
        "やっと着いたね。{timeword}からずっと待ってたよ。",
    ],
    "ask": [
        "それで、{noun_thing}どうだった?",
        "きみも{activity}ことある?",
        "今週末、{plan}のはどう?",
        "正直なところ、{food}って好き?",
        "{place}にはもう行きましたか?",
        "もし{person}に会えたら、なんて言う?",
        "どうして{abstract}がこんなに気になるんだろう。",
        "ちょっとだけ{music}の話をしてもいい?",
        "{timeword}は空いてる?だめならいいんだけど。",
        "最後に{feeling}になったのって、いつですか?",
    ],
    "react": [
        "うん、それは{adjpos}。反論は受け付けません。",
        "うわ、それは{adjneg}ね。",
        "え、ほんとに?それ{adjpos}じゃん。",
        "ほら、こういうところが好きなんだよ。",
        "{timeword}聞いた中でいちばん笑った。",
        "黙ってそれやったの、信じられない。",
        "正直に言うと、{adjpos}。それだけです。",
        "うーん、{adjneg}。そうなっちゃったの、残念だね。",
        "うそでしょ。{person}が本当にそう言ったの?",
        "それ、胸にきた。ごまかせないな。",
    ],
    "opinion": [
        "{abstract}って、もっと評価されていいと思う。",
        "わたしにとっては、{noun_thing}はいまだに効くんだよね。",
        "今なら{food}のためにだいたい何でも差し出せる。",
        "{music}は、うまく言えない何かをくれるんです。",
        "{activity}のは立派なセルフケアだと決めました。",
        "誰もいない{place}に勝るものはないよ。",
        "{small_thing}を捨てられなくて、ずっと持ってる。",
        "{weather}が一日を変える力、みんな軽く見すぎです。",
        "わたしに言わせれば、もっと{plan}べきだよ。",
        "{abstract}の話、あんまりされない気がする。",
    ],
    "story": [
        "{timeword}、{activity}のに夢中で時間を忘れてた。",
        "よりによって{place}で{person}に会ったんだ。",
        "{weather}が来て、通り全体が急に静かになった。",
        "古いコートのポケットから{small_thing}が出てきたの。",
        "結局{place}で、とんでもない時間に{food}を食べてた。",
        "{person}がカードゲームを教えようとして、途中であきらめた。",
        "停電しちゃって、ただ{music}を聞いてた。",
        "午後はずっと{activity}のに使ったけど、まったく後悔してない。",
        "誰かが玄関に{small_thing}を置いていった。手紙もなしで。",
        "{place}に着いた瞬間に雨が降り出したんだ。",
    ],
    "quote": [
        "彼女はこっちを見て、「{quote}」って言ったの。",
        "そしたら{person}が「{quote}」だって。それだけ。",
        "雨の日になると母が「{quote}」って言ってたのを思い出す。",
        "「{quote}」。届いたのはそれだけだった。",
        "メモにはただ「{quote}」と書いてあった。",
        "{person}は肩をすくめて「{quote}」と言いました。",
        "あの一言がずっと残ってる。「{quote}」。",
        "きみが前に「{quote}」って言ったこと、忘れてないよ。",
        "最後の言葉は「{quote}」でした。",
        "彼は笑って「{quote}」って、当たり前みたいに言った。",
    ],
    "formal": [
        "こんばんは。よい一日を過ごされましたか。",
        "お待たせして申し訳ありません。件の問題は解決いたしました。",
        "僭越ながら、{abstract}はもっと注目されるべきだと思います。",
        "ご返信が遅れましたこと、お詫び申し上げます。",
        "ご都合がよろしければ、明日{noun_thing}についてお話しできますか。",
        "またお話しできて光栄です。",
        "{noun_thing}について簡単にまとめておきました。",
        "「{adjpos}」では言い足りないくらいです。",
        "{timeword}のご都合はいかがでしょうか。",
        "考え直したのですが、季節が変わる前に{plan}ほうがよいかと思います。",
    ],
    "close": [
        "とにかく、ちゃんと休んでね。",
        "はい、もう黙ります。たぶん。",
        "どうなったか、あとで教えて。",
        "帰ってくるまでここにいるよ。",
        "よし、行っておいで。きみなら大丈夫。",
        "おやすみ。あんまり考えすぎないで。",
        "またね。お水飲んでね。",
        "わたしからは以上です。いってらっしゃい。",
        "家に着いたら連絡してね。",
        "はい、わたしの話はここまで。次はきみの番。",
    ],
}

LANGS = {
    "en": {"pools": EN_POOLS, "templates": EN_TEMPLATES, "join": " "},
    "id": {"pools": ID_POOLS, "templates": ID_TEMPLATES, "join": " "},
    "ja": {"pools": JA_POOLS, "templates": JA_TEMPLATES, "join": ""},
}

SLOT_RE = re.compile(r"\{(\w+)\}")
CASE_RE = re.compile(r"""(^["']?|[.!?,:]\s+["']|[.!?]\s+)([a-z])""")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def unit_count(text: str, lang: str) -> int:
    """Length in whatever unit is meaningful for the language."""
    if lang == "ja":
        return len(re.sub(r"\s+", "", text))
    return len(text.split())


def render(template: str, pools: dict, rng: random.Random, used: set) -> str:
    """Fill {slot} placeholders, avoiding repeats within one utterance."""

    def pick(match: re.Match) -> str:
        pool = pools[match.group(1)]
        value = rng.choice(pool)
        for _ in range(8):
            if value not in used:
                break
            value = rng.choice(pool)
        used.add(value)
        return value

    return SLOT_RE.sub(pick, template)


def capitalize_sentence(text: str) -> str:
    """Upper-case every sentence start, including template-internal ones.

    Templates often open with a slot ("{person} shrugged...") or contain their
    own sentence break ("Oh, hi. {noun_thing} stuck in my head"), and pool
    phrases are stored lower-case so they can also appear mid-sentence.
    """
    return CASE_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def build_sentences(lang: str, shape: list[str], rng: random.Random) -> list[str]:
    spec = LANGS[lang]
    used_values: set = set()
    used_templates: set = set()
    sentences = []

    for category in shape:
        bank = spec["templates"][category]
        # A shape can repeat a category (formal/formal, story/story); reusing the
        # same template twice in one utterance reads like a stutter.
        template = rng.choice(bank)
        for _ in range(8):
            if template not in used_templates:
                break
            template = rng.choice(bank)
        used_templates.add(template)

        sentence = render(template, spec["pools"], rng, used_values)
        if lang != "ja":
            sentence = capitalize_sentence(sentence)
        sentences.append(sentence)

    return sentences


def apply_emotion_tag(sentences: list[str], lang: str, rng: random.Random) -> None:
    """Insert one tag at a clause start, in place."""
    index = rng.randrange(len(sentences))
    sentence = sentences[index]
    tag = rng.choice(EMOTION_TAGS)
    separator = "" if lang == "ja" else " "

    # Clause-internal placement, but never right before an opening quote: a tag
    # wedged between "said," and the quotation reads as part of the quote.
    comma = "、" if lang == "ja" else ", "
    positions = [
        m.end()
        for m in re.finditer(re.escape(comma), sentence)
        if m.end() < len(sentence) and sentence[m.end()] not in '"「'
    ]
    if positions and rng.random() < 0.35:
        cut = rng.choice(positions)
        sentence = sentence[:cut] + tag + separator + sentence[cut:]
    else:
        sentence = tag + separator + sentence

    sentences[index] = sentence


def make_text(lang: str, length_class: str, tagged: bool, rng: random.Random) -> str:
    """One utterance. Short utterances are resampled until they hit the band."""
    low, high = SHORT_BAND[lang]
    join = LANGS[lang]["join"]

    attempts = 40 if length_class == "short" else 1
    best: list[str] = []
    for _ in range(attempts):
        shape = rng.choice(SHAPES[length_class])
        sentences = build_sentences(lang, shape, rng)
        if length_class != "short":
            best = sentences
            break
        count = unit_count(join.join(sentences), lang)
        if low <= count <= high:
            best = sentences
            break
        if not best or unit_count(join.join(sentences), lang) < unit_count(
            join.join(best), lang
        ):
            best = sentences

    if tagged:
        apply_emotion_tag(best, lang, rng)

    return join.join(best)


def exact_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Split `total` across `mix` so the parts sum exactly to `total`."""
    counts = {key: int(total * share) for key, share in mix.items()}
    remainder = total - sum(counts.values())
    for key in sorted(mix, key=lambda k: -mix[k]):
        if remainder <= 0:
            break
        counts[key] += 1
        remainder -= 1
    return counts


def seed_for(utt_id: str) -> int:
    digest = hashlib.sha256(utt_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def build(count: int, rng_seed: int) -> list[dict]:
    rng = random.Random(rng_seed)

    # Exact quotas rather than per-draw probabilities: 2500 samples is small
    # enough that sampling drift would show up in the per-language totals.
    plan: list[tuple[str, str, bool]] = []
    for lang, lang_count in exact_counts(count, LANG_MIX).items():
        for length_class, class_count in exact_counts(lang_count, LENGTH_MIX).items():
            plan.extend([(lang, length_class, False)] * class_count)

    tag_targets = rng.sample(range(len(plan)), int(round(len(plan) * TAG_RATE)))
    for index in tag_targets:
        lang, length_class, _ = plan[index]
        plan[index] = (lang, length_class, True)

    hot = int(round(count * HOT_TEMPERATURE_RATE))
    temperatures = [HOT_TEMPERATURE] * hot + [COLD_TEMPERATURE] * (count - hot)
    rng.shuffle(temperatures)
    rng.shuffle(plan)

    seen: set[str] = set()
    collisions = 0
    records = []
    for index, (lang, length_class, tagged) in enumerate(plan):
        text = make_text(lang, length_class, tagged, rng)
        for _ in range(50):
            if text not in seen:
                break
            text = make_text(lang, length_class, tagged, rng)
        else:
            collisions += 1
        seen.add(text)

        utt_id = f"u{index + 1:06d}"
        records.append(
            {
                "utt_id": utt_id,
                "text": text,
                "lang": lang,
                "reference_id": REFERENCE_CYCLE[index % len(REFERENCE_CYCLE)],
                "temperature": temperatures[index],
                "seed": seed_for(utt_id),
                "length_class": length_class,
                "has_emotion_tag": tagged,
            }
        )

    if collisions:
        click.echo(f"warning: {collisions} utterances remained duplicates")
    return records


def report(records: list[dict]) -> None:
    langs = Counter(r["lang"] for r in records)
    classes = Counter(r["length_class"] for r in records)
    refs = Counter(r["reference_id"] for r in records)
    temps = Counter(r["temperature"] for r in records)
    tagged = sum(r["has_emotion_tag"] for r in records)
    total = len(records)

    click.echo(f"utterances       : {total}")
    click.echo(f"unique texts     : {len({r['text'] for r in records})}")
    click.echo(
        "language         : "
        + ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in sorted(langs.items()))
    )
    click.echo(
        "length class     : "
        + ", ".join(
            f"{k}={classes[k]} ({classes[k] / total:.1%})"
            for k in ("short", "medium", "long")
        )
    )
    click.echo(f"emotion tags     : {tagged} ({tagged / total:.1%})")
    click.echo(
        "reference_id     : " + ", ".join(f"{k}={v}" for k, v in sorted(refs.items()))
    )
    click.echo(
        "temperature      : " + ", ".join(f"{k}={v}" for k, v in sorted(temps.items()))
    )

    for lang in sorted(langs):
        unit = "chars" if lang == "ja" else "words"
        for length_class in ("short", "medium", "long"):
            lengths = [
                unit_count(r["text"], lang)
                for r in records
                if r["lang"] == lang and r["length_class"] == length_class
            ]
            if not lengths:
                continue
            lengths.sort()
            click.echo(
                f"  {lang}/{length_class:<6} n={len(lengths):<5} "
                f"min={lengths[0]:<4} p50={lengths[len(lengths) // 2]:<4} "
                f"max={lengths[-1]:<4} {unit}"
            )


@click.command()
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_OUT)
@click.option("--count", type=int, default=DEFAULT_COUNT)
@click.option("--seed", type=int, default=DEFAULT_SEED)
@click.option("--samples", type=int, default=0, help="print N example utterances")
def main(out: Path, count: int, seed: int, samples: int) -> None:
    records = build(count, seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    click.echo(f"wrote {out}")
    report(records)

    if samples:
        click.echo("--- samples ---")
        span = max(samples - 1, 1)
        picks = {round(k * (len(records) - 1) / span) for k in range(samples)}
        for record in (records[i] for i in sorted(picks)):
            click.echo(
                f"[{record['lang']}/{record['length_class']}] "
                f"{record['reference_id']} t={record['temperature']} "
                f"{record['text']}"
            )


if __name__ == "__main__":
    main()
