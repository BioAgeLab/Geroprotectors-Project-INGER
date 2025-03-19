# Cargar librerías necesarias
library(shiny)
library(readr)
library(dplyr)
library(stringi)
library(DT)
library(promises)
library(shinythemes)
library(rsconnect)


# Cargar la base de datos
geroprotectores <- read.csv("data/New_geroprotectors.csv")

# Convertir imágenes a datos binarios
logo_inger_b <- base64enc::dataURI(file = "www/logo_inger_b.jpg", mime = "image/jpeg")
abstract_geros <- base64enc::dataURI(file = "www/abstract_geros.jpg", mime = "image/jpeg")

# Definir la UI
ui <- fluidPage(
  theme = shinytheme("cosmo"),  # Cambiar el tema a "cosmo" para un diseño más moderno
  titlePanel(
    div(
      img(src = logo_inger_b, height = "60px", style = "float:left; margin-right:10px;"),
      "Database of compounds identified as possible geroprotectants by Machine-Learning"
    )
  ),
  
  tabsetPanel(
    # Primera pestaña: Búsqueda y Diagrama de Venn
    tabPanel("Search compound",
             sidebarLayout(
               sidebarPanel(
                 textInput("search", "Search compound:", ""),
                 actionButton("searchBtn", "Search", class = "btn-primary"),  # Botón con estilo primario
                 br(), br(),
                 h4("Instructions for Use"),
                 p("You can search for a particular compound by entering its name as reported by the PubChem database (https://pubchem.ncbi.nlm.nih.gov/) and clicking the \u201cSearch\u201d button.")
               ),
               mainPanel(
                 img(src = abstract_geros, height = "400px", style = "display: block; margin-left: auto; margin-right: auto;"),  # Centrar la imagen
                 br(),
                 DTOutput("search_results")  # Resultados de búsqueda en tabla
               )
             )
    ),
    
    # Segunda pestaña: Base de datos completa
    tabPanel("Database",
             fluidRow(
               column(12, 
                      downloadButton("download_data", "Download", class = "btn-success"),  # Botón con estilo de éxito
                      br(), br(),
                      DTOutput("full_table")
               )
             )
    ),
    
    # Tercera pestaña: Resumen
    tabPanel("Information",
             h3("Project"),
             p("Aging is a substrate for the development of many chronic diseases worldwide. Age-related diseases and syndromes pose a growing challenge to healthcare systems worldwide, resulting in poor quality of life and poor health in the later stages of life. Intervening in the molecular effects of aging can prevent or delay the onset of many diseases. In this context, geroprotectants aim to regulate the underlying molecular mechanisms of these age-related diseases with a single pharmacological intervention, as demonstrated in animal models. Therefore, machine learning and chemoinformatics can be used to search for new compounds with geroprotective properties. This study analysed the Geroprotectors database containing many compounds with geroprotective activity. Three Machine Learning models, Decision Tree Classifier (DTC), Support Vector Machine (SVM) and Nearest Neighbors (KNN), were built to predict whether or not compounds have desired geroprotective effects, using chemical descriptors calculated from the chemical structure of each compound as features. We applied our model to predict compounds with possible geroprotective activity in the Coconut database, where the geroprotective activity of natural compounds hosted in this database is unknown.")
    )
  )
)

# Definir la lógica del servidor
server <- function(input, output, session) {
  
  # Filtrar resultados de búsqueda
  output$search_results <- renderDT({
    req(input$searchBtn)
    isolate({
      search_query <- input$search
      resultados <- geroprotectores[grep(search_query, geroprotectores$Name, ignore.case = TRUE), ]
      datatable(resultados, options = list(pageLength = 5, scrollX = TRUE))  # Opciones para la tabla
    })
  })
  
  # Mostrar toda la base de datos
  output$full_table <- renderDT({
    datatable(geroprotectores, options = list(pageLength = 10, scrollX = TRUE))  # Opciones para la tabla
  })
  
  # Botón para descargar la base de datos
  output$download_data <- downloadHandler(
    filename = function() {
      paste("geroprotectors_database", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(geroprotectores, file, row.names = FALSE)
    }
  )
}

# Ejecutar la aplicación Shiny
shinyApp(ui = ui, server = server)
