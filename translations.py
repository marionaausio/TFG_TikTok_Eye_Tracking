# -*- coding: utf-8 -*-
"""Participant-facing UI strings for the experiment, in English / Spanish / Catalan.

This is the SINGLE place to review/edit participant wording. Operator-facing dialogs
(EEG QC, gaze calibration, preflight, stream-health) deliberately stay English and
are NOT here.

  SCREENS - full-screen text blocks, keyed by name
  PROMPTS - the question prompt for each demographic/rating item key
  LABELS  - answer-button labels, keyed by the ENGLISH label (numeric labels like the
            Juster 0-10 / interest 1-7 are language-neutral and fall through unchanged).
            Label text is shared across ALL scales by design: an identical English label
            maps to the same translation regardless of which question it belongs to
            (verified safe for the live scales). If a future scale ever needs a
            context-specific wording for a reused English word, key LABELS by
            (item_key, english_label) and pass it.key into label().

Spanish = peninsular (Spain); informal address (tu) to match a young study sample.
NOTE: the Spanish and Catalan values were checked before final data collection.
off, especially the 7-point liking/virality/familiarity gradients (the instrument).
"""

LANGUAGES = [("en", "English"), ("es", "Español"), ("ca", "Català")]


def screen(key, lang):
    return SCREENS.get(lang, {}).get(key) or SCREENS["en"].get(key, key)


def prompt(key, lang):
    return PROMPTS.get(lang, {}).get(key) or PROMPTS["en"].get(key, key)


def label(en_label, lang):
    # Labels absent from a language table (numbers, percentages) pass through unchanged.
    return LABELS.get(lang, {}).get(en_label) or en_label


SCREENS = {
    "en": {
        "welcome": "Welcome to the Experiment\n\nPress SPACE to continue",
        "demographics_title": "Before we begin - a few questions about you",
        "demographics_subtitle": "Type your age, then click one answer per row (you may leave any row blank). Then click CONTINUE.",
        "age_prompt": "How old are you?",
        "continue_button": "CONTINUE",
        "targeting_practice": ("Practice - Mouse Target Test\n\n"
            "First, a short PRACTICE round to get you used to it.\n\n"
            "Focus on the cross in the centre of the screen.\n"
            "When the circle appears, move the mouse and click inside it as FAST and ACCURATELY as possible.\n"
            "DO NOT click the fixation cross - only click inside the circle.\n"
            "The mouse cursor resets to the centre at the start of each trial.\n\n"
            "Press SPACE to start the practice."),
        "targeting_transition": ("Good - that was practice.\n\n"
            "Now the real test begins.\n\n"
            "Press SPACE when you are ready."),
        "gonogo_rule": ("New rule - read carefully!\n\n"
            "Circles will appear as before, but now in TWO shades:\n\n"
            "   GREY circle  ->  click the circle  (same as before)\n"
            "   WHITE circle ->  do NOT click it - click the  +  in the centre instead\n\n"
            "The  +  stays in the middle the whole time.\n"
            "Go as FAST as you can while staying accurate.\n\n"
            "First a short practice. Press SPACE to start."),
        "gonogo_transition": ("Good - that was practice.\n\n"
            "Now the real test begins.\n\n"
            "Remember:  GREY -> click the circle,  WHITE -> click the centre  +\n\n"
            "Press SPACE when you are ready."),
        "rating_title": "Your impression of this video",
        "rating_subtitle": "Click a label for each row, or leave it blank. Then click CONTINUE.",
        "video_block_intro": ("Video Block\n\n"
            "You will now watch the videos with sound.\n"
            "Please keep your gaze on the screen and remain still.\n\n"
            "Press SPACE to continue."),
        "end_screen": "Thank you for participating in the experiment.\n\nPress SPACE to end.",
    },
    "es": {
        "welcome": "Bienvenido/a al experimento\n\nPulsa ESPACIO para continuar",
        "demographics_title": "Antes de empezar, unas preguntas sobre ti",
        "demographics_subtitle": "Escribe tu edad y haz clic en una respuesta por fila (puedes dejar filas en blanco). Luego haz clic en CONTINUAR.",
        "age_prompt": "¿Cuántos años tienes?",
        "continue_button": "CONTINUAR",
        "targeting_practice": ("Práctica - Prueba de puntería con el ratón\n\n"
            "Primero, una ronda de PRÁCTICA breve para que te acostumbres.\n\n"
            "Fija la mirada en la cruz del centro de la pantalla.\n"
            "Cuando aparezca el círculo, mueve el ratón y haz clic dentro de él lo más RÁPIDO y PRECISO que puedas.\n"
            "NO hagas clic en la cruz de fijación, solo dentro del círculo.\n"
            "El cursor vuelve al centro al inicio de cada intento.\n\n"
            "Pulsa ESPACIO para empezar la práctica."),
        "targeting_transition": ("Bien, eso era práctica.\n\n"
            "Ahora empieza la prueba de verdad.\n\n"
            "Pulsa ESPACIO cuando estés listo/a."),
        "gonogo_rule": ("Nueva regla, ¡lee con atención!\n\n"
            "Aparecerán círculos como antes, pero ahora en DOS tonos:\n\n"
            "   círculo GRIS    ->  haz clic en el círculo  (igual que antes)\n"
            "   círculo BLANCO  ->  NO hagas clic - haz clic en el  +  del centro\n\n"
            "El  +  se queda en el centro todo el rato.\n"
            "Ve lo más RÁPIDO que puedas sin perder precisión.\n\n"
            "Primero una práctica breve. Pulsa ESPACIO para empezar."),
        "gonogo_transition": ("Bien, eso era práctica.\n\n"
            "Ahora empieza la prueba de verdad.\n\n"
            "Recuerda:  GRIS -> clic en el círculo,  BLANCO -> clic en el  +  del centro\n\n"
            "Pulsa ESPACIO cuando estés listo/a."),
        "rating_title": "Tu impresión de este vídeo",
        "rating_subtitle": "Haz clic en una etiqueta por fila, o déjala en blanco. Luego haz clic en CONTINUAR.",
        "video_block_intro": ("Bloque de vídeos\n\n"
            "Ahora verás los vídeos con sonido.\n"
            "Mantén la mirada en la pantalla y quédate quieto/a.\n\n"
            "Pulsa ESPACIO para continuar."),
        "end_screen": "Gracias por participar en el experimento.\n\nPulsa ESPACIO para terminar.",
    },
    "ca": {
        "welcome": "Benvingut/da a l'experiment\n\nPrem ESPAI per continuar",
        "demographics_title": "Abans de començar - unes quantes preguntes sobre tu",
        "demographics_subtitle": "Escriu la teva edat i clica una resposta per fila (pots deixar qualsevol fila en blanc). Després clica CONTINUA.",
        "age_prompt": "Quants anys tens?",
        "continue_button": "CONTINUA",
        "targeting_practice": ("Pràctica - Prova de precisió amb el ratolí\n\n"
            "Primer faràs una ronda curta de pràctica per familiaritzar-te amb la tasca.\n\n"
            "Mira la creu del centre de la pantalla.\n"
            "Quan aparegui el cercle, mou el ratolí i fes clic a dins tan ràpidament i amb tanta precisió com puguis.\n"
            "No facis clic a la creu de fixació: fes clic només dins del cercle.\n"
            "El cursor tornarà al centre a l'inici de cada assaig.\n\n"
            "Prem ESPAI per començar la pràctica."),
        "targeting_transition": ("Molt bé, has acabat la pràctica.\n\n"
            "Ara començarà la prova.\n\n"
            "Prem ESPAI quan estiguis a punt."),
        "gonogo_rule": ("Nova norma - llegeix-la amb atenció!\n\n"
            "Apareixeran cercles com abans, però ara en DOS tons:\n\n"
            "   cercle GRIS  ->  clica el cercle  (igual que abans)\n"
            "   cercle BLANC ->  NO el cliquis - clica el  +  del centre\n\n"
            "El  +  es queda al mig tota l'estona.\n"
            "Vés tan RÀPID com puguis sense perdre precisió.\n\n"
            "Primer una pràctica curta. Prem ESPAI per començar."),
        "gonogo_transition": ("Bé - això era la pràctica.\n\n"
            "Ara comença el test de veritat.\n\n"
            "Recorda:  GRIS -> clica el cercle,  BLANC -> clica el  +  del centre\n\n"
            "Prem ESPAI quan estiguis a punt."),
        "rating_title": "La teva impressió d'aquest vídeo",
        "rating_subtitle": "Clica una etiqueta per cada fila, o deixa-la en blanc. Després clica CONTINUA.",
        "video_block_intro": ("Bloc de vídeos\n\n"
            "Ara veuràs els vídeos amb so.\n"
            "Mantén la mirada a la pantalla i queda't quiet/a.\n\n"
            "Prem ESPAI per continuar."),
        "end_screen": "Gràcies per participar en l'experiment.\n\nPrem ESPAI per acabar.",
    },
}

PROMPTS = {
    "en": {
        "gender": "What is your gender?",
        "handedness": "Which hand do you use for the mouse?",
        "vision_correction": "Are you wearing glasses or contact lenses right now?",
        "first_language": "Which language are you most comfortable in?",
        "tiktok_use": "How often do you use TikTok?",
        "cosmetics_buy": "How often do you buy cosmetics or make-up?",
        "cosmetics_interest": "How interested are you in cosmetics / make-up?   (1 = not at all, 7 = very)",
        "seen_video_before": "Have you seen this video before?",
        "liking": "How did you feel about this video?",
        "purchase_intent": "How likely are you to buy this product?   (0 = not at all, 10 = definitely would)",
        "virality": "Do you think this video went viral on TikTok?",
        "brand_familiarity": "How familiar are you with this brand?",
    },
    "es": {
        "gender": "¿Cuál es tu género?",
        "handedness": "¿Con qué mano usas el ratón?",
        "vision_correction": "¿Llevas gafas o lentillas ahora mismo?",
        "first_language": "¿En qué lengua te sientes más cómodo/a?",
        "tiktok_use": "¿Con qué frecuencia usas TikTok?",
        "cosmetics_buy": "¿Con qué frecuencia compras cosméticos o maquillaje?",
        "cosmetics_interest": "¿Cuánto te interesan los cosméticos / el maquillaje?   (1 = nada, 7 = mucho)",
        "seen_video_before": "¿Habías visto este vídeo antes?",
        "liking": "¿Qué te ha parecido este vídeo?",
        "purchase_intent": "¿Qué probabilidad hay de que compres este producto?   (0 = ninguna, 10 = seguro que sí)",
        "virality": "¿Crees que este vídeo se hizo viral en TikTok?",
        "brand_familiarity": "¿Hasta qué punto conoces esta marca?",
    },
    "ca": {
        "gender": "Quin és el teu gènere?",
        "handedness": "Amb quina mà fas servir el ratolí?",
        "vision_correction": "Ara mateix portes ulleres o lentilles?",
        "first_language": "En quina llengua et sents més còmode/a?",
        "tiktok_use": "Cada quant fas servir TikTok?",
        "cosmetics_buy": "Cada quant compres cosmètics o maquillatge?",
        "cosmetics_interest": "Quin interès tens pels cosmètics / el maquillatge?   (1 = gens, 7 = molt)",
        "seen_video_before": "Havies vist aquest vídeo abans?",
        "liking": "Què t'ha semblat aquest vídeo?",
        "purchase_intent": "Quina probabilitat hi ha que compris aquest producte?   (0 = cap, 10 = segur que sí)",
        "virality": "Creus que aquest vídeo es va fer viral a TikTok?",
        "brand_familiarity": "Fins a quin punt coneixes aquesta marca?",
    },
}

LABELS = {
    "en": {},   # English labels are the keys themselves (pass-through).
    "es": {
        "Female": "Mujer", "Male": "Hombre", "Other": "Otro",
        "Prefer\nnot to say": "Prefiero\nno decirlo",
        "Right": "Derecha", "Left": "Izquierda", "Either": "Cualquiera",
        "Neither": "Ninguna", "Glasses": "Gafas", "Contacts": "Lentillas",
        "Spanish": "Castellano", "Catalan": "Catalán", "English": "Inglés",
        "Daily": "A diario", "A few/\nweek": "Varias/\nsemana", "Weekly": "Semanal",
        "Rarely": "Casi nunca", "Never": "Nunca", "Monthly": "Mensual",
        "Every\nfew mo.": "Cada\nvarios m.",
        "Strongly\ndisliked": "Me ha\ndisgustado\nmucho", "Disliked": "Me ha\ndisgustado",
        "Slightly\ndisliked": "Me ha\ndisgustado\nun poco", "Neutral": "Neutral",
        "Slightly\nliked": "Me ha\ngustado\nun poco", "Liked": "Me ha\ngustado",
        "Strongly\nliked": "Me ha\nencantado",
        "Definitely\nNOT viral": "Seguro\nque NO", "Probably\nnot": "Probablemente\nno",
        "Maybe\nnot": "Quizá\nno", "Unsure": "No lo sé", "Maybe\nviral": "Quizá\nsí",
        "Probably\nviral": "Probablemente\nsí", "Definitely\nviral": "Seguro\nque sí",
        "Not at all": "Nada", "Slightly": "Un poco", "Somewhat": "Algo",
        "Moderately": "Bastante", "Quite": "Mucho", "Very": "Muchísimo",
        "Extremely": "Al máximo",
        "No": "No", "Not sure": "No estoy\nseguro/a", "Yes": "Sí",
    },
    "ca": {
        "Female": "Dona", "Male": "Home", "Other": "Altre",
        "Prefer\nnot to say": "Prefereixo\nno dir-ho",
        "Right": "Dreta", "Left": "Esquerra", "Either": "Qualsevol",
        "Neither": "Cap", "Glasses": "Ulleres", "Contacts": "Lentilles",
        "Spanish": "Castellà", "Catalan": "Català", "English": "Anglès",
        "Daily": "Cada dia", "A few/\nweek": "Uns quants/\nsetmana", "Weekly": "Cada\nsetmana",
        "Rarely": "Gairebé\nmai", "Never": "Mai", "Monthly": "Cada mes",
        "Every\nfew mo.": "Cada\npocs mes.",
        "Strongly\ndisliked": "M'ha\ndesagradat\nmolt", "Disliked": "M'ha\ndesagradat",
        "Slightly\ndisliked": "M'ha\ndesagradat\nuna mica", "Neutral": "Neutre",
        "Slightly\nliked": "M'ha\nagradat\nuna mica", "Liked": "M'ha\nagradat",
        "Strongly\nliked": "M'ha\nagradat\nmolt",
        "Definitely\nNOT viral": "Segur que\nNO viral", "Probably\nnot": "Probablement\nno",
        "Maybe\nnot": "Potser\nno", "Unsure": "No ho sé", "Maybe\nviral": "Potser\nviral",
        "Probably\nviral": "Probablement\nviral", "Definitely\nviral": "Segur que\nviral",
        "Not at all": "Gens", "Slightly": "Molt poc", "Somewhat": "Una mica",
        "Moderately": "Moderadament", "Quite": "Força", "Very": "Molt",
        "Extremely": "Moltíssim",
        "No": "No", "Not sure": "No\nho sé", "Yes": "Sí",
    },
}
