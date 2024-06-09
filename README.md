# Image_Classification-CIFAR-10

Este relatório descreve o desenvolvimento de um modelo de classificação de imagens utilizando o conjunto de dados CIFAR-10, API Flask e Ngrok. O modelo permite aos usuários enviar imagens para serem classificadas em uma das 10 categorias do CIFAR-10: avião, automóvel, pássaro, gato, veado, cachorro, sapo, cavalo, navio e caminhão. Para fazer o uso de tal modelo, é possível fazer o download do arquivo cifar10.ipynb acima e usar as imagens disponíveis ou qualquer outra de acordo com a preferência. Neste relatório será abordado a construção do modelo, como configurar e usar.

## CIFAR-10
Antes de partir para a construção e características da implementação, é válido destacar o uso do CIFAR-10. O CIFAR-10 é um conjunto de dados distribuídas em 10 classes diferentes, cada uma com 6.000 imagens. As classes do CIFAR-10 vão desde animais até objetos como automóveis, cada uma sendo exclusiva. Seu conjunto de dados é divido em duas partes: treinamento e teste. O conjunto de treinamento contém 50.000 imagnes, com 5.000 imagens de cada classe. Já o conjunto de teste possui 10.000 imagens, com 1.000 por classe. Esse tipo de divisão auxilia no treino e teste dos modelos, auxiliando também na avaliação de desempenho em relação a generalização dos dados. Além disso, é válido lembrar que as imagens do CIFAR-10 são em resolução baixa.

## Importações necessárias
* drive;
* tensorflow;
* keras do tensorflow;
* joblib;
* numpy.
### Importações para API
* flask & flask - ngrok;
* pyngrok - ngrok;
* threading;
* Pil;
* io;
* Requests.

## Modelo
O modelo utilizado é uma rede convolucional genérica com arquitetura CNN. Este modelo é composto pelas seguintes características:

* Camadas convolucionais:
    * Três camadas convolucionais, sendo as três com função de ativação ReLU. A primeira camada possui 64 filtros de tamanho 5x5 e entrada de imagens de 32x32 com três canais RGB. As outras duas camadas convolucionais também possui função de ativação ReLU, porém, as duas tem 128 filtros de tamanho 5x5.

* MaxPooling(Pooling máximo):
    * Há duas camadas de pooling máximo, ambas são com tamanho 2x2, o qual reduz pela metade as dimensões espaciais da saída da camada convolucional

* Flatten e camada densa:
    * Flatten achata a saída da última camada convolucional para transformar os dados de uma matriz 3D para um vetor 1D;
    * A camada densa tem 128 unidades com a ativação ReLU e a camada de saída tem 10, com função de ativação softmax para classificação multiclasse.

Ordem:
1. Primeira Camada Convolucional: 64 filtros 5x5 com ativação ReLU, entrada de imagens de 32x32 pixels com 3 canais(RGB);
2. Primeira Camada de Pooling: Pooling máximo com uma janela de 2x2;
3. Segunda Camada Convolucional: 128 filtros de tamanho 5x5 com ativação ReLU;
4. Segunda Camada de Pooling: Pooling máximo com uma janela de 2x2;
5. Terceira Camada Convolucional: 128 filtros de tamanho 5x5 com ativação ReLU;
6. Flatten;
7. Camada Densa: 128 unidades com ativação ReLU;
8. Camada de Saída: 10 unidades com ativação softmax.

### Compilação do modelo
* Otimizador: otimizador Adam, ajusta automaticamente a taxa de aprendizado;
* Função de Perda: SparseCategoricalCrossentropy com from_logits=True, indicando que a função de perda espera saídas em forma de logits;
Métrica: Acurácia ou "accuracy", para avaliar o desempenho do modelo durante o treino e a validação.

### Treinamento do modelo
* Config: treinado com 5 épocas com um tamanho de lote de 64;
* Dados de Validação: utilizados para ajustar os parâmetros e evitar o sobreajuste;
* Callback: interrompe o treinamento se não houver melhoria na loss de validação após 2 épocas e restaura os melhores pesos encontrados.

### Salvamento do modelo com .h5 e uso
O modelo é salvo no arquivo "model.h5" que armazena a arquitetura completa, os pesos treinados e o estado do otimizador, permitindo retomar o treinamento.

## Impelmentação da API
Nesta seção, será mostrada a implementação da API Flask junto ao ngrok para rodar uma aplicação no servidor a fim de permitir que o usuário faça uma requisição POST de uma imagem e receber a classificação desta.

### Configuração e instalação
#### Flask e ngrok
O Flask é utilizado para criar uma aplicação web simples, enquanto o ngrok é usado para criar um túnel público que expõe a aplicação local à Internet. Isso é particularmente útil em ambientes de desenvolvimento sem um endereço IP público, como o Google Colab.

Instalação de Pacotes: Os pacotes necessários incluem flask-ngrok para integrar Flask com ngrok e pyngrok para gerenciar túneis ngrok. Também é realizada uma atualização do pyngrok para garantir que a versão mais recente esteja em uso.

Para usar definitivamente o ngrok, é necessário efetuar cadastro e login na plataforma, a plataforma permite o uso de apenas três túneis de forma gratuita, para implementar no modelo e iniciar um servidor, é necessário obter um token gerado no momento que é feito o cadastro na plataforma, para isso, é necessário entrar em sua conta, ir em dashboard e na seção chamada "your authtoken" e copiar o token disponível e inserir no seu projeto. Além disso, é preciso ter o Google Authenticator associado a sua conta o qual controla cada sessão de uso com segurança.

* Instruções de funcionamento:
    1. Acessar o Site do ngrok:
        Vá para ngrok e clique em "Sign up" para criar uma nova conta. Você pode se registrar usando seu endereço de e-mail ou contas de redes sociais como Google ou GitHub;

    2. Preencher Informações:
        Insira as informações necessárias, como seu e-mail e senha. Se você escolher usar uma conta de rede social, algumas informações podem ser preenchidas automaticamente;

    3. Confirmar E-mail:
        Após o registro, você precisará verificar seu endereço de e-mail. Verifique sua caixa de entrada e clique no link de confirmação enviado pelo ngrok;

    4. Instalar Google Authenticator:
        Baixe e instale o aplicativo Google Authenticator em seu dispositivo móvel, disponível na Google Play Store ou na Apple App Store;

    5. Escanear o Código QR:
        Use o Google Authenticator para escanear o código QR fornecido pelo ngrok durante o processo de configuração da verificação em duas etapas;

    6. Inserir Código de Verificação:
        Após escanear o código QR, o aplicativo exibirá um código de 6 dígitos. Digite esse código no site do ngrok para completar a configuração da verificação em duas etapas;
    
    7. Acessar o Dashboard DO Ngrok após login:
        No painel de controle do ngrok, procure por uma seção chamada "Auth" ou "Your Auth Token";
    
    8. Copiar o AuthToken:
        Você verá seu token de autenticação nesta seção. Clique em "Copy" para copiar o token;
    
    9. Configurar o token no seu projeto:
        Use a seguinte configuração substituindo o "SEU_TOKEN_AQUI" pelo token copiado: 
            from pyngrok import ngrok
            ngrok.set_auth_token("SEU_TOKEN_AQUI")

#### Recuperação e Desconexão de Túneis:
 O script verifica os túneis ngrok ativos e os desconecta se necessário, o que é útil para evitar limitações no número de túneis simultâneos permitidos na versão gratuita do ngrok o qual já foi mencionado anteriormente. Cso você rode seu modelo várias vezes, provavelmente esse limite irá se exceder, por isso essa etapa é essencial.

 ### Aplicação Flask
 * Inicialização da Aplicação: Uma instância do Flask é criada e configurada para usar ngrok;
 * Carregamento do Modelo: O modelo é carregado do disco, é praticamente uma replicagem do modelo CNN construído anteriormente que está armazenado em '/content/model.h5';
 * Rota de Predição: É definida uma rota /predict que aceita apenas requisições POST. Esta rota lida com o recebimento de imagens, processamento e classificação das mesmas usando o modelo carregado;
 * Processamento de Imagens: As imagens recebidas são abertas, redimensionadas para 32x32 pixels, convertidas para um array numpy, normalizadas e preparadas para predição;
 * Execução da Aplicação: A aplicação Flask é executada em uma thread separada, permitindo que outras operações sejam realizadas simultaneamente sem bloquear o servidor, sendo muito útil no projeto feito no Google Colab, visto que a célula que roda o Flask não irá bloquear as outras, então enquanto o servidor Flask está ativo, as outras células podem ser rodadas tranquilamente.

 #### Requisição da predição
 * Requisição POST: Envia imagens no formato png ou jpg à API Flask usando requisições POST através do pacote requests importado;
 * URL do ngrok: A URL gerada pelo ngrok é usada para fazer as requisições POST, permitindo o acesso externo a API;
 * Envio e Resposta: A imagem inputada é enviada para a API, após isso, a resposta é enviada, incluindo a classe prevista que é mostrada seguindo a numeração especifica da classe do CIFAR-10.

 ### Classes do CIFAR-10 numeradas:
 Normalmente o output vem em formato JSON e com a numeração especifica que representa cada classe, a seguir é possível verificar cada classe e sua respectiva numeração:

    0. Avião (Airplane)
    1. Carro (Automobile)
    2. Pássaro (Bird)
    3. Gato (Cat)
    4. Veado (Deer)
    5. Cachorro (Dog)
    6. Sapo (Frog)
    7. Cavalo (Horse)
    8. Navio (Ship)
    9. Caminhão (Truck)

 ### Casos de teste e resultados
 Foram feito quatro tipos de inputs com quatro animais diferentes e todos tiveram uma resposta correta pelo modelo. Os animais foram gato, cachorro, sapo e cavalo.

* Caso 1:

    Gato: classe 3
    
    1/1 [==============================] - 0s 28ms/step
    INFO:werkzeug:127.0.0.1 - - [09/Jun/2024 04:43:10] "POST /predict HTTP/1.1" 200 -
    {"predicted_class":"3"}

    imagem: ![Gato](/img/down-syndrome-cat.jpg)

* Caso 2:

    Cachorro: classe 5

    1/1 [==============================] - 0s 115ms/step
    INFO:werkzeug:127.0.0.1 - - [09/Jun/2024 04:41:31] "POST /predict HTTP/1.1" 200 -
    {"predicted_class":"5"}

    imagem: ![Cachorro](/img/hd-aspect-1500566326-gettyimages-512366437-1.jpg)

* Caso 3:

    Sapo: classe 6
    
    1/1 [==============================] - 0s 26ms/step
    INFO:werkzeug:127.0.0.1 - - [09/Jun/2024 04:46:04] "POST /predict HTTP/1.1" 200 -
    {"predicted_class":"6"}

    imagem: ![Sapo](/img/Facebook+Cover+Photo+Your+Favourite+Vet.jpg)

* Caso 4:

    Cavalo: classe 7

    1/1 [==============================] - 0s 30ms/step
    INFO:werkzeug:127.0.0.1 - - [09/Jun/2024 16:27:36] "POST /predict HTTP/1.1" 200 -
    {"predicted_class":"7"}

    imagem: 

    ![Cavalo](/img/thumb-horse.jpg)


A acurácia do modelo na última época(5) teve o valor de 0.7053, sendo a loss de 0.8357, val_loss de 0.9252 e val_accuracy de 0.6779, tendo todas as predições efetuadas com sucesso e corretamente como pode ser visto nos casos de teste acima.
