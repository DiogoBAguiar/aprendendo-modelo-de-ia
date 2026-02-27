#from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

frases = [
    "Estou feliz hoje", # alegria
    "Que dia maravilhoso", # alegria
    "Estou radiante de felicidade", # alegria
    "Sinto uma alegria imensa agora", # alegria
    "Meu coração está leve e contente", # alegria
    "Estou animado com a vida", # alegria
    "Isso me deixa muito satisfeito", # alegria
    "Estou sorrindo à toa", # alegria
    "Que momento incrível", # alegria
    "Estou em paz e feliz", # alegria
    "Nada pode estragar meu bom humor", # alegria
    "Estou grato por tudo", # alegria
    "Hoje é um dia perfeito", # alegria
    "Estou cheio de esperança", # alegria
    "Meu dia está fantástico", # alegria
    "Estou empolgado com o futuro", # alegria
    "Sinto-me realizado", # alegria
    "Estou muito contente", # alegria
    "Minha vida está ótima", # alegria
    "Estou celebrando cada instante", # alegria
    "Estou me sentindo triste", # Tristeza
    "Não quero mais viver", # Tristeza
    "Sinto um vazio enorme", # Tristeza
    "Estou profundamente abalado", # Tristeza
    "Nada parece fazer sentido", # Tristeza
    "Estou chorando por dentro", # Tristeza
    "Meu coração está pesado", # Tristeza
    "Não vejo saída para isso", # Tristeza
    "Sinto-me sozinho", # Tristeza
    "Perdi a vontade de tudo", # Tristeza
    "Estou desanimado com a vida", # Tristeza
    "Tudo está dando errado", # Tristeza
    "Sinto uma dor constante", # Tristeza
    "Estou decepcionado comigo", # Tristeza
    "Nada me faz sorrir", # Tristeza
    "Sinto que fracassei", # Tristeza
    "Estou esgotado emocionalmente", # Tristeza
    "Queria desaparecer", # Tristeza
    "Meu mundo desmoronou", # Tristeza
    "Estou sem esperança", # Tristeza
    "Estou com muita raiva", # Raiva
    "Isso me deixa furioso", # Raiva
    "Estou irritado demais", # Raiva
    "Não suporto essa situação", # Raiva
    "Estou perdendo a paciência", # Raiva
    "Isso é injusto", # Raiva
    "Estou revoltado com tudo", # Raiva
    "Quero gritar de ódio", # Raiva
    "Estou indignado", # Raiva
    "Isso me tira do sério", # Raiva
    "Estou explodindo por dentro", # Raiva
    "Não aceito isso", # Raiva
    "Estou extremamente frustrado", # Raiva
    "Isso é absurdo", # Raiva
    "Estou cheio de rancor", # Raiva
    "Minha tolerância acabou", # Raiva
    "Estou irritado com você", # Raiva
    "Isso me enfurece", # Raiva
    "Estou cansado dessa palhaçada", # Raiva
    "Vou perder o controle", # Raiva
    "Estou com medo", # Medo
    "Isso me assusta", # Medo
    "Estou apavorado", # Medo
    "Sinto um frio na barriga", # Medo
    "Tenho receio do que vai acontecer", # Medo
    "Estou inseguro", # Medo
    "Isso parece perigoso", # Medo
    "Estou tremendo de medo", # Medo
    "Tenho medo de falhar", # Medo
    "Estou preocupado", # Medo
    "Isso me dá ansiedade", # Medo
    "Estou nervoso com isso", # Medo
    "Tenho medo do futuro", # Medo
    "Estou em pânico", # Medo
    "Isso me deixa vulnerável", # Medo
    "Estou assustado com a situação", # Medo
    "Sinto que algo ruim vai acontecer", # Medo
    "Tenho medo de perder tudo", # Medo
    "Estou apreensivo", # Medo
    "Isso me causa tensão", # Medo
    "Estou surpreso", # Surpresa
    "Isso foi inesperado", # Surpresa
    "Não estava esperando por isso", # Surpresa
    "Que novidade incrível", # Surpresa
    "Estou impressionado", # Surpresa
    "Isso me pegou desprevenido", # Surpresa
    "Uau, que surpresa", # Surpresa
    "Estou chocado", # Surpresa
    "Isso mudou tudo de repente", # Surpresa
    "Não acredito no que vi", # Surpresa
    "Isso foi totalmente inesperado", # Surpresa
    "Estou admirado", # Surpresa
    "Que reviravolta", # Surpresa
    "Estou boquiaberto", # Surpresa
    "Isso é inacreditável", # Surpresa
    "Fiquei sem palavras", # Surpresa
    "Que acontecimento surpreendente", # Surpresa
    "Estou maravilhado", # Surpresa
    "Isso superou minhas expectativas", # Surpresa
    "Não sei o que dizer", # Surpresa
    
]

# 0: Alegria, 1: Tristeza, 2: Raiva, 3: Medo, 4: Surpresa

y = [
    # Alegria: Frases 1 a 20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    
    # Tristeza: Frases 21 a 40
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    
    # Raiva: Frases 41 a 60
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    
    # Medo: Frases 61 a 80
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    
    # Surpresa: Frases 81 a 100
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
]

vectorizer = CountVectorizer()
x =  vectorizer.fit_transform(frases).toarray()

print("Palavras únicas:", vectorizer.get_feature_names_out())
print(x)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regressaoLinear = LogisticRegression()

regressaoLinear.fit(x_train, y_train)

y_pred = regressaoLinear.predict(x_test)

acuracia = accuracy_score(y_test, y_pred)
print("Acurácia:", acuracia)