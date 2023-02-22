# Importamos librerías

import streamlit as st 
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly_express as px

# Configuración de página :

st.set_page_config(page_title= "EDA Forbes Global 2000 & Forbes Billonarios 2022", layout= "centered" )

# Leemos csv:

df_2000=pd.read_csv(r"C:\VSCode\samplerepo\Proyecto Final\forbesglobal2000_final.csv")
df_bill=pd.read_csv(r"C:\VSCode\samplerepo\Proyecto Final\forbesbillonarios_final.csv")

# APP :
    
st.image('dataset-cover.jpg', width=1000, use_column_width=True)
st.title("EDA 'Forbes Global 2000' & 'Forbes billonarios 2022' ")
st.text ("Análisis de los listados publicados por Forbes en 2022")

col1, col2= st.columns(2)

with col1:
    st.write ("¿Cuáles son las empresas más poderosas?")

with col2:
    st.write('¿Qué países dominan la economía global?')


# SIDEBAR :

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("Proyecto Final Data Analytics")
st.sidebar.write(" Elena Pindado ")
st.sidebar.image("forbes.jpg", width=170, caption='info:www.forbes.es')
st.sidebar.write("---") # hará un salto de línea 


if st.sidebar.button("Forbes Global 2000"):
    tabs=st.tabs(["Contenido de los datos","Overview", "Identificación Outliers", "Distribución de los datos"])

    tab_plots= tabs[0]
    with tab_plots:
     st.write("Forbes Global 2000 clasifica a las empresas más grandes del mundo utilizando cuatro métricas: ventas, ganancias, activos y valor de mercado. Se han utilizado los datos financieros de los últimos 12 meses disponibles al 22 de abril de 2022 para calcular las métricas utilizadas para su clasificación.  A continuación, el listado de las 2000 empresas : ")
     st.dataframe(df_2000) 

    tab_plots= tabs[1]
    with tab_plots:
     st.write("Overview: visión general de nuestros datos")    
     import pandas_profiling
     from streamlit_pandas_profiling import st_profile_report


     df = pd.read_csv(r"C:\VSCode\samplerepo\Proyecto Final\forbes global 2022(2000 companies).csv")
     pr = df.profile_report()

     st_profile_report(pr)

    tab_plots= tabs[2]
    with tab_plots:
        st.write("Lo veremos a través de un gráfico boxplot : nos ayudará a identificar patrones y tendencias que de otra manera pueden pasar desapercibidos. Además, nos pueden ayudar a encontrar valores atípicos que pueden estar influenciando nuestros resultados .")
        st.image('outliers.png')
        st.write(" 1. Podemos identificar valores atípicos: Valores atípicos bajos (negativos) en la variable 'profit': beneficios que se alejan de la distribución normal de los datos y que representan pérdidas, por lo que ya sabemos que algunas de las empresas de este listado tienen pérdidas en beneficios. ")
        st.write("2. Valores atípicos superiores , muy altos en 'assets': activos, con muchos valores que se alejan desmesuradamente de la media de los datos, ocurriendo lo mismo en 'market value' : valor de mercado, con algunas empresas con estos valores muy por encima de la media. * Estos valores son más altos que los valores de la mayoría de la muestra y son un indicador de que hay una mayor variación en los datos y no se consideran como parte de la distribución normal de los datos.") 

    tab_plots= tabs[3]
    with tab_plots:
        st.write(" Vamos a verlo a través de un gráfico displot : nos servirá para explorar la distribución  de los datos , con las variables numéricas . Esto nos ayudará a identificar patrones y tendencias. Veremos la cantidad de veces que se repite un valor en las variables 'sales', 'profit', 'assets' y 'market value'. Esto nos permitirá ver si hay valores que se repiten con frecuencia o si hay algunos valores que son extremadamente infrecuentes.")
        sns.distplot(df_2000['sales'], color='blue')
        st.pyplot()
        st.write("Podemos observar como no hay una distribución normal de los datos , algunos valores en ventas muy por encima del resto de empresas. ")
        sns.distplot(df_2000['profit'], color='m')
        st.pyplot()
        st.write("En beneficios podemos ver como hay datos distribuidos tanto hacia valores negativos, que representan pérdidas, como valores muy altos por encima de la media.")
        sns.distplot(df_2000['assets'], color='r')
        st.pyplot()
        st.write("Como ocurrió con 'sales' podemos ver una distribución normal de los datos y algunos valores que se alejan de la distribución meida de los datos.")
        sns.distplot(df_2000['market value'], color='g')
        st.pyplot()
        st.write("Comprobamos también una distribución normal de los datos.")

if st.sidebar.button("Data Preprocessing"):
    tabs=st.tabs(["Limpieza de datos ","Transformación de datos"])

    tab_plots= tabs[0]
    with tab_plots:
     st.write("Comprobamos si tenemos valores nulos en nuestros datos :")
     st.code(
        """df_2000.isnull().sum()"""
     )
     st.code(
        """df_bill.isnull().sum()"""
     )
     st.write("Eliminamos columnas que no aportan información relevante :")
     st.code(
        """df_bill = df_bill.drop(columns=['Unnamed: 0'])"""
     )

    tab_plots= tabs[1]
    with tab_plots:
     st.write(" Creamos un nuevo índice con la columna 'rank' :")
     st.code(
        """df_2000 = df_2000.set_index("rank ")"""
     )
     st.code(
        """df_bill = df_bill.set_index("rank")"""
     )
     st.write(" Tenemos valores en billones y millones , por lo que unificamos las unidades a través de una función:  ")
     st.code(
        """def convert(x): 
        # función que transforme nuestros datos a numéricos y unifique las unidades a billones, además elimine '$ ' y ','
    x=x.replace('$','')
    x=x.replace(',','')
    if 'M' in x :
         x=x.replace('M','')
         x=pd.to_numeric(x)
         x=x/1000
    else:
        x=x.replace('B','')
        x=pd.to_numeric(x)
        
    return x"""
      )
     st.write("Eliminamos símbolo $ y B (billones) en los datos:")
     st.code(
        """df_bill['networth'] = df_bill['networth'].str.replace('$','').str.replace('B','')"""
      )

if st.sidebar.button( "Relación entre variables"):
    tabs=st.tabs([" Distribución general de los datos","Relación 'Valor de mercado' ","Correlación variables","Correlación Pearson/Spearman"])
    
    tab_plots= tabs[0]
    with tab_plots:
        st.write("Distribución general de los datos Global 2000 : ")
        st.image('distribuciongeneral.png')

    tab_plots= tabs[1]
    with tab_plots:
     st.write("Usaremos un gráfico regplot:  sirve para visualizar la relación entre dos variables. Esto nos permitirá ver si existe una correlación entre las variables y si hay algún patrón o tendencia que se pueda ver claramente. También veremos una línea de regresión, lo que nos permitirá predecir los valores futuros para una variable usando los valores de la otra.")
     st.write(" Vamos a investigar la variable valor de mercado 'market value' y como puede este crecer en un futuro.")
     st.write("Analizamos la relación entre las ventas y el valor de mercado: ")
     sns.regplot(x="sales", y="market value", data=df_2000, color="m")
     st.pyplot()
     st.write(" Como podemos observar, hay una relación positiva entre las dos variables, a medida que aumentan las ventas aumenta el valor de mercado de las empresas. La mayoría de las empresas distribuyen uniformamente sus ventas , con valores fuera de rango en agunas de ellas.")
     st.write("Analizamos la relación entre los activos y el valor de mercado: ")  
     sns.regplot(x="assets", y="market value", data=df_2000, color='g')  
     st.pyplot()
     st.write("Hay una tendencia positiva en los activos en relación con el valor de mercado, como indica la línea de regresión. La mayoría de las empresas tienen una distribución normal de los activos, con algunos valores atípicos de forma muy puntual.")
     st.write("La relación es más fuerte entre las ventas y el valor de mercado, que entre los activos y el valor de mercado, como lo indica la pendiente más pronunciada en la línea de regresión.")    

    tab_plots= tabs[2]
    with tab_plots:
       st.write("Vamos a ver ahora como se relacionan las variables ventas, beneficios y activos con la variable 'valor de mercado' con un gráfico de dispersión pairplot:  podremos ver la correlación entre estas variables e identificar patrones en los datos. ")
       st.image("distribucion2.png")
       st.write("Podemos observar que todas estas variables tienen una alta relación con 'valor de mercado', tiene sentido pensar que a medida que aumentan las ventas, los activos y los beneficios lo hace también el valor de mercado de las empresas.")
 
    tab_plots= tabs[3]
    with tab_plots:
        st.write("Realizamos ahora un gráfico 'mapa de calor' aplicando los métodos de correlación de Pearson y Spearman para medir las relaciones entre las variables y 'valor de mercado'. ")
        st.write("Una correlación positiva (1) significa que a medida que aumenta una variable, también aumenta la otra. Una correlación negativa (-1) significa que a medida que aumenta una variable, la otra disminuye.")    
        st.image("correlacion.png")
        st.write("¿ Qué nos indica Pearson?")
        st.write(" Todas las variables están relacionadas de forma positiva. Las variables 'profit' y 'market value' , están altamente correlacionadas de forma positiva (0.80) , cuando una aumenta lo hace también la otra: cuando aumentan las ventas aumenta también su valor de mercado y es donde se da el mayor grado de correlación; siendo donde menos (0.15) entre las variables 'assets' y 'market value'.")
        st.write("¿Qué nos indica Spearman? ")
        st.write("Todas las variables están relacionadas de forma positiva. Vemos la correlación más alta y positiva entre las variables profit' y 'market value', es decir, a medida que aumentan los beneficios aumenta también el  valor de mercado de la empresa; mientras que la correlación con menor grado la vemos entre 'assets y 'market value'. ")


if st.sidebar.button("Test/ Modelo"):
    tabs=st.tabs(["Test de normalidad", "Modelo Supervisado Regresión"])


    tab_plots= tabs[0]
    with tab_plots:
        st.write("Vamos a comprobar si nuestros datos provienen de una distribución Gaussiana (Normal).")
        st.write("Usaremos para ello el Test de Shapiro-Wilk:")
        st.write("¿Qué debemos asumir?")
        st.write(" - Que las observaciones de las muestras son Independientes e Idénticamente Distribuidas (idd)")
        st.write("Tomaremos la variable nuevamente de 'valor de mercado' : ")
        st.code(
        '''
        shapiro_test = shapiro(df_2000['market value'])

        print(f"El p-valor obtenido en el test de Shapiro-Wilk es de {shapiro_test[1]}")")
        ''')
        st.write("El p-valor obtenido en el test de Shapiro-Wilk es de 0.0")
        st.write("Un p-valor de 0.0 significa que los datos no se ajustan a una distribución normal. Esto significa que los datos de la variable 'market value' no se distribuyen de forma simétrica alrededor de una media, por tanto la variable 'market value' no se puede usar para predecir los resultados de procesos aleatorios.")
        st.write("Mostramos los resultados de este test en un grafico QQ plot: ")
        st.image("qqplot.png")
        st.write("Para que la distribución fuera normal la línea azul debería estar sobre la roja, aquí no lo está, como podemos observar en el gráfico.")


    tab_plots= tabs[1]
    with tab_plots:
      st.write("Vamos a realizar una regresión múltiple con un modelo de regresión lineal : es un modelo que se utiliza para predecir una variable dependiente (y) a partir de una o más variables independientes (x). ")
      st.write("Primero vamos a definir las variables independientes (X) y dependientes (y). En este caso, vamos a estudiar las variables independientes  'profit' y 'sales' y la variable dependiente es 'market value', de esta forma prodremos predecir como crecerá el valor de mercado o no en relación con las ventas y los beneficios.")
      st.code(
      '''
      # Dividimos el dataset en dos partes: entrenamiento y pruebas:
      X = df_2000[['profit', 'sales']]
      y = df_2000['market value']

      # Dividimos los datos en entrenamiento y test:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Instaciamos el modelo
      model = LinearRegression()

      # Entrenamos el modelo
      model.fit(X_train, y_train)

      # Hacer predicciones con el modelo entrenado
      y_pred = model.predict(X_test)

      # Calculamos el error cometido por las predicciones:
      mse = mean_squared_error(y_test, y_pred) # error cuadrático medio
      rmse = np.sqrt(mse) # calculamos la raíz cuadrada del error cuadrático medio (MSE)
      mae = mean_absolute_error(y_test, y_pred)

      # Muestra los errores
      print('MSE: ', mse)
      print('RMSE: ', rmse)
      print('MAE: ', mae)
      ''')
      st.write("MSE:  1515.5407461731731")
      st.write("RMSE:  38.929946650017044")
      st.write("MAE:  22.384579110975796")
      st.write("MSE (Error Cuadrático Medio): Mide la diferencia entre los valores predichos y los valores reales. Es una medida de la precisión del modelo. Cuanto más bajo sea el MSE, mejor será el modelo.")
      st.write("RMSE (Raiz del Error Cuadrático Medio): Es una medida de la desviación promedio entre los valores predichos y los valores reales. Cuanto más bajo sea el RMSE, mejor será el modelo.")
      st.write("MAE (Error Absoluto Medio): Mide la diferencia promedio entre los valores predichos y los valores reales. Cuanto más bajo sea el MAE, mejor será el modelo.")
      st.write("De los resultados obtenidos, se puede concluir que el modelo de regresión lineal es bastante preciso, ya que los valores MSE, RMSE y MAE son relativamente bajos. Esto significa que el modelo es capaz de predecir los valores reales con una buena precisión.")
      st.write("Vamos a ver que nos indica otra métrica, 'R-squared' :")
      st.code(
      ''' score = model.score(X_test, y_test)
      print("R-squared: ", score)
      ''')   
    
      st.write("R-squared:  0.31247496458818813")
      st.write("R-squared:  0.31, nos indica que el modelo explica el 31 % de la variación en los datos. Esto sugiere que el modelo puede mejorar si se ajustan los parámetros, se agregan más características o se prueban otros modelos.")

if st.sidebar.button("Análisis Global 2000"):
    tabs=st.tabs(["Top mejores empresas", "Top empresas por ventas ","Top empresas beneficios","Top empresas valor mercado"])
      
    tab_plots= tabs[0]
    with tab_plots:
      st.write("Las 10 empresas más poderosas del mundo: ")
      st.dataframe(df_2000.head(10)) 
      st.write("El multimillonario holding empresarial Berkshire Hathaway, propiedad de Warren Buffet, lleva desde 2019 en este top 10. El primer puesto lo obtiene de acciones en más de una veintena de grandes empresas multinacionales como The Coca-Cola Company, American Express, General Electric, Kraft foods, Johnson & Johnson, y Heinz.")

    tab_plots= tabs[1]
    with tab_plots:
      st.write("Top 10 empresas por ventas: ")
      top_10_sales = df_2000.nlargest(10, 'sales')
      grafico=px.bar(top_10_sales, x='global company', y='sales', color='global company', title='Top 10 empresas por ventas')
      st.plotly_chart(grafico)
      st.write("Walmart es la mayor empresa del mundo por facturación. Sus ventas sumaron 600.000 millones de dólares en el último año. Solo hay 21 países en el mundo que tengan un PIB superior .")

    tab_plots= tabs[2]
    with tab_plots:
      st.write("Top 10 empresas por beneficios: ")
      top_10_profit = df_2000.nlargest(10, 'profit')
      grafico=px.bar(top_10_profit, x='global company', y='profit', color='global company', title='Top 10 empresas por beneficios')
      st.plotly_chart(grafico)
      st.write("Como muestra nuestro gráfico, Saudi Aramco fue la empresa cotizada más rentable del planeta en 2022. El año pasado, la compañía estatal de petróleo y gas de Arabia Saudí, que salió a bolsa a finales de 2019, obtuvo unas ganancias de unos 105.400 millones de dólares, las más altas del mundo .")

    tab_plots= tabs[3]
    with tab_plots:
      st.write("Top 10 empresas por valor de mercado: ")
      top_10_value = df_2000.nlargest(10, 'market value')
      grafico=px.bar(top_10_value, x='global company', y='market value', color='global company', title='Top 10 empresas por valor de mercado')
      st.plotly_chart(grafico)
      st.write("Apple se sitúa en primera posición con un valor de marca de 482.215 millones de dólares, gracias a un incremento del 18 % experimentado respecto a la edición anterior del informe.")

if st.sidebar.button("Análisis por países"):
    tabs=st.tabs(["Top 10 países", "Empresas por países ","Empresas españolas","Porcentaje empresas españolas"])

    tab_plots= tabs[0]
    with tab_plots:
      st.write("¿ En qué países se encuentran las empresas del top 10?")
      import plotly.graph_objects as go
      fig = go.Figure()
      fig.add_trace(go.Bar(x=df_2000['country'][:10], y=df_2000['global company'][:10], text=df_2000['global company'][:10], textposition='auto', marker_color=['red', 'green', 'blue', 'orange', 'purple', 
                                                       'yellow', 'pink', 'brown', 'magenta', 'darkblue']))
      st.plotly_chart(fig) 
      st.write("Estados Unidos, China, Arabia Saudi y Japón concentran el top 10 de las empresas más poderosas del mundo.")

    tab_plots= tabs[1]
    with tab_plots:
      st.write("¿ Qué países tienen mayor cantidad de empresas en este listado?: ")
      st.image("toppaises.png")
      st.write("Estados Unidos concentra la mayor cantidad de empresas compitiendo con países asiáticos donde se concentra la mayor cantidad de empresas :China, Japón y Corea del Sur. ")

    tab_plots= tabs[2]
    with tab_plots:
      st.write("¿Cuántas empresas españolas hay en este listado?")
      st.dataframe(df_2000[df_2000['country']=='Spain'])
      st.write("Hay 21 empresas españolas en el listado. ")
    
    tab_plots= tabs[3]
    with tab_plots:
      st.write("¿Qué porcentaje representa España en este listado?")
      num_spain_companies = df_2000[df_2000['country']=='Spain'].shape[0]
      percent_spain_companies = num_spain_companies / df_2000.shape[0] * 100
      fig = px.pie(values=[percent_spain_companies, 100-percent_spain_companies], 
             names=['Spain', 'Other Countries'], 
             title='Porcentaje de empresas españolas en el listado')
      st.plotly_chart(fig)
      st.write("España representa un 1.05 % en este listado, y por tanto , en el mundo. ")

if st.sidebar.button("Análisis España"):
    tabs=st.tabs(["Empresas por valor mercado", "Empresas por beneficios ","Empresas por ventas"])
    
    tab_plots= tabs[0]
    with tab_plots:
        st.write("¿Qué empresas españolas tienen más valor de mercado? ")
        spain_global=df_2000[df_2000['country']=='Spain']
        top_10_spain_valor = spain_global.nlargest(10, 'market value')
        grafico=px.bar(top_10_spain_valor, x='global company', y='market value', color='global company', title='Top 10 empresas españolas por valor de mercado')
        st.plotly_chart(grafico)
        st.write("Iberdrola es la empresa española con mayor valor de mercado en el mundo.")

    tab_plots= tabs[1]
    with tab_plots:
        st.write("¿Qué empresas españolas tienen más beneficios? ")
        top_10_spain_benef = spain_global.nlargest(10, 'profit')
        grafico=px.bar(top_10_spain_benef, x='global company', y='profit', color='global company', title='Top 10 empresas españolas por beneficios')
        st.plotly_chart(grafico)
        st.write("El banco Santander es la empresa española con mayor beneficio en el mundo.")

    
    tab_plots= tabs[2]
    with tab_plots:
        st.write("¿Qué empresas españolas tienen mayor volumen de ventas? ")
        top_10_spain_sales = spain_global.nlargest(10, 'sales')
        grafico=px.bar(top_10_spain_sales, x='global company', y='sales', color='global company', title='Top 10 empresas españolas por ventas')
        st.plotly_chart(grafico)
        st.write("El banco Santander es la empresa española con mayor volumen de ventas del mundo.")

if st.sidebar.button("Caso Santander"):
    tabs=st.tabs(["Santander vs B.Hathaway por ventas ", "Santander vs B.Hathaway por activos "])

    tab_plots= tabs[0]
    with tab_plots:
      st.write("La empresa española que ocupa el primer puesto del ranking es Santander, en el puesto 58 del ranking mundial ,vamos a hacer una comparativa con la primera empresa del listado a nivel mundial: Berkshire Hathaway , comparando sus ventas: ")
      st.image("santander.png")
      st.write(" Podemos ver la enorme diferencia entre las ventas de las dos empresas. ")

    tab_plots= tabs[1]
    with tab_plots:
      st.write("Sin embargo, si comparamos por activos , Santander tiene casi el doble de activos que la primera empresa mundial, un dato muy interesante : ")
      st.image("santanderactivos.png") 

if st.sidebar.button("Forbes Billonarios"):
    tabs=st.tabs(["Top 10 billonarios", "Billonarios por países","Industrias billonarios","Top 10 empresas" ])

    tab_plots= tabs[0]
    with tab_plots:
      st.write("Los 2.668 multimillonarios del planeta valen 12,7 billones de dólares. ¿Quiénes ocupan los primeros puestos de esta lista? ¿Quiénes son los hombres más ricos del planeta? En general, 1.891 multimillonarios, o el 71% de la lista , se hicieron a sí mismos, lo que significa que fundaron o cofundaron una empresa o establecieron su propia fortuna (en lugar de heredarla).")
      st.write(" Top 10 billonarios comparando su patrimonio neto: ")
      st.image("billonarios.png")
      st.write("Elon Musk es el hombre más rico del mundo, encabezando esta lista por primera vez; programador, físico y empresario nacido en Sudáfrica, pero nacionalizado canadiense y norteamericano, fundador de empresas como Paypal, Tesla, SpaceX y Solar City. Podemos ver además, la diferencia de patrimonio con el segundo hombre más rico del mundo Jeff Bezos. Musk tenía un valor estimado de 219 mil millones de dólares (199,66 mil millones de euros), luego de añadir 68 mil millones de dólares (62 mil millones de euros) a su fortuna durante el año pasado gracias a un aumento del 33% en el precio de las acciones de su fabricante de vehículos eléctricos Tesla.")
    
    tab_plots= tabs[1]
    with tab_plots:
      st.write("¿ Qué países tienen más cantidad de billonarios?")
      st.image("paisesbillonarios.png")
      st.write("Estados Unidos tiene más multimillonarios que cualquier otro país, con 735, frente a los 724 del año pasado. China sigue en segundo lugar, con 607 (incluidos Hong Kong y Macao), seguida de India (166),Alemania (134) y Rusia (83).Las grandes pérdidas se producen en Rusia y China.La guerra de Ucrania, y la avalancha de sanciones que siguió, hizo que el mercado de valores ruso y el rublo se desplomaran, lo que resultó en 34 multimillonarios rusos menos en la lista. Llama poderosamente la atención como los ricos indios florecen entre la pobreza extrema.")

    tab_plots= tabs[2]
    with tab_plots:
      st.write("¿De dónde obtienen sus ingresos? ¿ A qué industrias pertenecen sus empresas?")
      st.image("industriasbillonarios.png")
      st.write("Finanzas e inversiones, tecnología, fabricación , moda y salud , son los sectores que más billonarios transforman.")

    tab_plots= tabs[3]
    with tab_plots:
      st.write("Top 10 empresas de los billonarios: ¿ Coinciden estas empresas con el listado top 10 de Global 2000?")
      st.image("empresasbillonarios.png")
      st.write("Podemos ver como dos de las empresas del Top 10 en Global 2000 : la primera Berkshire Hathaway y Amazon aparecen también en este ranking.")
    
if st.sidebar.button("Españoles billonarios"):
    tabs=st.tabs(["Españoles billonarios", "Porcentaje billonarios españoles","A.Ortega vs Elon Musk","A.Ortega vs otros" ]) 

    tab_plots= tabs[0]
    with tab_plots:
      st.write("¿Cuántos españoles hay en la lista ?")
      st.dataframe(df_bill[df_bill['country']=='Spain'])
      st.write(" Hay 26 españoles en la lista.")

    tab_plots= tabs[1]
    with tab_plots:
      st.write("¿Qué porcentaje de billonarios españoles hay en el mundo?")
      num_spain_bill = df_bill[df_bill['country']=='Spain'].shape[0]
      percent_spain_bill = num_spain_bill / df_bill.shape[0] * 100
      fig = px.pie(values=[percent_spain_bill, 100-percent_spain_bill], 
             names=['Spain', 'Other Countries'], 
             title='Porcentaje de billonarios españoles en el mundo')
      st.plotly_chart(fig)
      st.write(" España representa un 1 % de los billonarios de todo el mundo .")

    tab_plots= tabs[2]
    with tab_plots:
      st.write("El primer español que aparece en la lista es Amancio Ortega, vamos a hacer una comparativa de patrimonio entre él y el primer billonario de la lista mundial 'Elon Musk' : ")
      st.image("amanciovselon.png")
      st.write(" Podemos ver la enorme diferencia que hay entre los dos; El empresario gallego, fundador de Inditex, tiene un patrimonio estimado de 56.500 millones de dólares. Cae doce posiciones respecto a 2021, saliendo del top 10 , y ocupando el puesto 23.")
      
    tab_plots= tabs[3]
    with tab_plots:
      st.write("¿Y qué diferencia hay entre el primer español de la lista y el último?")
      st.image("españabillonarios.png")
      st.write(" Amancio Ortega ocupando el puesto 23 mundial, tiene una grandísima diferencia con el último español billonario de la lista : Jorge Gallardo Ballart, que ocupa el puesto 2578, casi al final del listado de Forbes, dedidado al sector farmacéutico. ")
    

if st.sidebar.button("Edades billonarios"):
    tabs=st.tabs(["Billonario más joven", "Distribución edades" ]) 

    tab_plots= tabs[0]
    with tab_plots:
      st.write("¿Quién es el billonario más joven del mundo?")
      st.code(
        """
        min_age = df_bill['age'].min()
        name = df_bill.loc[df_bill['age'] == min_age, 'name'].iloc[0]
        source = df_bill.loc[df_bill['age'] == min_age, 'source'].iloc[0]

        print('El billonario más joven es', name, 'con una edad de', min_age, 'y su empresa es', source)
        """)
      st.write("El billonario más joven es Kevin David Lehmann  con una edad de 19 y su empresa es Drugstore .Lo es gracias a su padre, que invirtió en una de las droguerías y parafarmacias más famosas de Alemania.")

    tab_plots= tabs[1]
    with tab_plots:
        st.write("¿ Qué edades tienen los billonarios?")
        st.image("edadesbillonarios.png")
        st.write("Podemos ver que la mayoría de los billonarios tienen más de 50 años, siendo mayoría los billonarios de más de 60, con casos excepcionales en menores de 30 años.")
      
if st.sidebar.button("Conclusiones"):
    tabs=st.tabs(["Conclusiones"])

    st.write("Para concluir, reafirmamos las preguntas de investigación y respondemos explícitamente según nuestros hallazgos:")
    st.write("¿Cuáles son las principales empresas clasificadas por ventas, beneficios y valor de mercado? ¿En qué países operan?")
    st.write("Enumeramos solo las 3 primeras:")
    st.write("-Ventas: Wal-Mart Stores (EE UU), Amazon (EE UU), Saudi Aramco (Arabia Saudí)")
    st.write("-Beneficios: Saudi Aramco (Arabia Saudí), Apple(EE UU), Berkshire Hathaway (EE UU)")
    st.write("-Valor de mercado: Apple (EE UU), Saudi Aramco (Arabia Saudí), Microsoft (EE UU)")
    st.write("Los sectores más representados en la clasificación de Forbes de este año fueron Banca, Finanzas diversificadas y Tecnología.")
    st.write("Los países que mayor poder económico concentran en el mundo entre empresas y billonarios son Estados Unidos y China, en una clara competición por dominar la economía mundial.")


   



