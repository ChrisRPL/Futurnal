"""Language sample fixtures for testing language detection accuracy.

Provides real multilanguage text samples for validating language detection
across different languages and scripts. All samples are authentic content
in the respective languages to ensure realistic testing conditions.
"""

# English sample
ENGLISH_SAMPLE = """
The quick brown fox jumps over the lazy dog. This is a test document written
in English to validate language detection capabilities. Natural language processing
has become increasingly important in modern applications, especially those dealing
with multilingual content and international audiences. The ability to accurately
detect and classify language enables better user experiences and more sophisticated
content analysis pipelines.
"""

# Spanish sample
SPANISH_SAMPLE = """
El rápido zorro marrón salta sobre el perro perezoso. Este es un documento de
prueba escrito en español para validar las capacidades de detección de idiomas.
El procesamiento del lenguaje natural se ha vuelto cada vez más importante en
las aplicaciones modernas, especialmente aquellas que tratan con contenido
multilingüe y audiencias internacionales. La capacidad de detectar y clasificar
con precisión el idioma permite mejores experiencias de usuario y canales de
análisis de contenido más sofisticados.
"""

# French sample
FRENCH_SAMPLE = """
Le rapide renard brun saute par-dessus le chien paresseux. Ceci est un document
de test écrit en français pour valider les capacités de détection de langue.
Le traitement du langage naturel est devenu de plus en plus important dans les
applications modernes, en particulier celles qui traitent du contenu multilingue
et des publics internationaux. La capacité de détecter et de classifier avec
précision la langue permet de meilleures expériences utilisateur et des pipelines
d'analyse de contenu plus sophistiqués.
"""

# German sample
GERMAN_SAMPLE = """
Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Testdokument
in deutscher Sprache zur Validierung der Spracherkennungsfähigkeiten. Die
Verarbeitung natürlicher Sprache ist in modernen Anwendungen zunehmend wichtig
geworden, insbesondere in solchen, die mit mehrsprachigen Inhalten und
internationalen Zielgruppen arbeiten. Die Fähigkeit, Sprache genau zu erkennen
und zu klassifizieren, ermöglicht bessere Benutzererfahrungen und ausgefeiltere
Content-Analyse-Pipelines.
"""

# Chinese sample (Simplified)
CHINESE_SAMPLE = """
快速的棕色狐狸跳过懒狗。这是一个用中文编写的测试文档,用于验证语言检测能力。
自然语言处理在现代应用程序中变得越来越重要,特别是那些处理多语言内容和国际
受众的应用程序。准确检测和分类语言的能力可以实现更好的用户体验和更复杂的
内容分析管道。机器学习和人工智能的进步使得语言检测系统能够以高精度处理
多种语言和方言。
"""

# Italian sample
ITALIAN_SAMPLE = """
La volpe marrone veloce salta sopra il cane pigro. Questo è un documento di test
scritto in italiano per convalidare le capacità di rilevamento della lingua.
L'elaborazione del linguaggio naturale è diventata sempre più importante nelle
applicazioni moderne, in particolare quelle che si occupano di contenuti
multilingue e pubblici internazionali. La capacità di rilevare e classificare
con precisione la lingua consente migliori esperienze utente e pipeline di
analisi dei contenuti più sofisticate.
"""

# Portuguese sample
PORTUGUESE_SAMPLE = """
A rápida raposa marrom pula sobre o cachorro preguiçoso. Este é um documento de
teste escrito em português para validar as capacidades de detecção de idioma.
O processamento de linguagem natural tornou-se cada vez mais importante em
aplicações modernas, especialmente aquelas que lidam com conteúdo multilíngue
e audiências internacionais. A capacidade de detectar e classificar com precisão
o idioma permite melhores experiências do usuário e pipelines de análise de
conteúdo mais sofisticados.
"""

# Russian sample
RUSSIAN_SAMPLE = """
Быстрая коричневая лиса прыгает через ленивую собаку. Это тестовый документ,
написанный на русском языке для проверки возможностей определения языка.
Обработка естественного языка стала все более важной в современных приложениях,
особенно тех, которые работают с многоязычным контентом и международной
аудиторией. Способность точно определять и классифицировать язык обеспечивает
лучший пользовательский опыт и более сложные конвейеры анализа контента.
"""

# Japanese sample
JAPANESE_SAMPLE = """
速い茶色のキツネが怠け者の犬を飛び越えます。これは言語検出機能を検証するために
日本語で書かれたテストドキュメントです。自然言語処理は、特に多言語コンテンツや
国際的な聴衆を扱うアプリケーションにおいて、現代のアプリケーションでますます
重要になっています。言語を正確に検出して分類する能力により、より良いユーザー
エクスペリエンスとより洗練されたコンテンツ分析パイプラインが可能になります。
"""

# Arabic sample
ARABIC_SAMPLE = """
الثعلب البني السريع يقفز فوق الكلب الكسول. هذا مستند اختبار مكتوب باللغة
العربية للتحقق من قدرات اكتشاف اللغة. أصبحت معالجة اللغة الطبيعية ذات أهمية
متزايدة في التطبيقات الحديثة، وخاصة تلك التي تتعامل مع المحتوى متعدد اللغات
والجماهير الدولية. تتيح القدرة على اكتشاف اللغة وتصنيفها بدقة تجارب مستخدم
أفضل وخطوط أنابيب تحليل محتوى أكثر تطوراً.
"""

# Mixed language sample (English + Spanish code-switching)
MIXED_LANGUAGE_SAMPLE = """
This is an interesting document that switches between languages. Por ejemplo,
aquí estoy escribiendo en español, but then I switch back to English. This kind
of code-switching is very common in multilingual communities, especialmente en
lugares donde se hablan múltiples idiomas. Language detection systems need to
handle these situations gracefully, aunque puede ser un desafío técnico.
"""

# Very short text (should return None)
SHORT_TEXT = "Hi"

# Empty text
EMPTY_TEXT = ""

# Language sample registry for easy access in tests
LANGUAGE_SAMPLES = {
    "en": ENGLISH_SAMPLE,
    "es": SPANISH_SAMPLE,
    "fr": FRENCH_SAMPLE,
    "de": GERMAN_SAMPLE,
    "zh": CHINESE_SAMPLE,
    "it": ITALIAN_SAMPLE,
    "pt": PORTUGUESE_SAMPLE,
    "ru": RUSSIAN_SAMPLE,
    "ja": JAPANESE_SAMPLE,
    "ar": ARABIC_SAMPLE,
    "mixed": MIXED_LANGUAGE_SAMPLE,
    "short": SHORT_TEXT,
    "empty": EMPTY_TEXT,
}


def get_sample(language_code: str) -> str:
    """Get language sample by code.

    Args:
        language_code: ISO 639-1 language code or 'mixed'/'short'/'empty'

    Returns:
        Text sample in the specified language

    Raises:
        KeyError: If language code not found
    """
    return LANGUAGE_SAMPLES[language_code]


def get_all_samples() -> dict:
    """Get all language samples.

    Returns:
        Dictionary mapping language codes to samples
    """
    return LANGUAGE_SAMPLES.copy()
