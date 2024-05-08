import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from EA_class import EvolutionaryAlgorithm
import webbrowser
import flask
import os

# Initialize the Dash app with Bootstrap support for better styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# for the best list of solution
best_solution = []

# globally Maintining the EA list
EA = EvolutionaryAlgorithm(population_size=100, mutation_rate=0.6, num_generations=5000, eta_c=1.0)

# EA = EvolutionaryAlgorithm(population_size=100, mutation_rate=0.6, num_generations=5000, eta_c = 1.0)
selected_foods = []


# Define the layout of the app
app.layout = dbc.Container([
    # dbc.Row(dbc.Col(html.H1("Diet Meal Plan Website", className="text-center my-4"), width=12)),

    dbc.Row([
            dbc.Col([
                html.H1("Athlete Meal Plan", className="text-center my-4"),
                html.H4([
                    "Helping you achieve your fitness goals!",
                    html.Span(" 🍽️", style={"vertical-align": "middle"})
                ], className="text-center")
            ], width=12, style={"color": "white", "background-color": "#242526"})
        ]),
        # Add an empty row to create space between the sections
        dbc.Row(html.Div(style={'height': '30px'})),


    dbc.Row([
        # Athlete Info Section
        dbc.Col(html.Div([
            html.H3("Athlete Info", className="bg-light p-3 text-center"),
            html.Label("Please select your sport:"),
            dcc.Dropdown(id='sport-dropdown', options=[
                {'label': 'Boxing', 'value': 'Boxers'},
                {'label': 'Running', 'value': 'Runners'},
                {'label': 'Swimming', 'value': 'Swimmers'},
                {'label': 'Cycling', 'value': 'Cyclists'},
                {'label': 'Football', 'value': 'Football'},
                {'label': 'Cricket', 'value': 'Cricket'},
                {'label': 'Tennis', 'value': 'Tennis'},
                {'label': 'Volleyball', 'value': 'Volleyball'},
                {'label': 'Basketball', 'value': 'Basketball'},
                {'label': 'Hockey', 'value': 'Hockey'},
                {'label': 'Formula1 Racer', 'value': 'Formula1 Racer'},
                {'label': 'Badminton', 'value': 'Badminton'},
                {'label': 'Rugby', 'value': 'Rugby'},
                {'label': 'Golf', 'value': 'Golf'},
            ], placeholder='Select a sport'),
            html.Label("Please select your age group:"),
            dcc.Dropdown(id='age-dropdown', options=[
                {'label': '20-30', 'value': '20-30'},
                {'label': '30-40', 'value': '30-40'},
                {'label': '40-50', 'value': '40-50'},
                {'label': 'Over 50', 'value': '50+'},
            ], placeholder='Select an age group'),
            html.Label("Please select your gender:"),
            dcc.Dropdown(id='gender-dropdown', options=[
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'},
            ], placeholder='Select a gender'),
        ], className="mb-5 p-3 border shadow"), width=6),


        # Food Preferences Section
        dbc.Col(html.Div([
            html.H3("Food Preferences", className="bg-light p-3 text-center"),
            html.Label("Please select your choice of Food:"),
            dcc.Dropdown(id='poultry-dropdown', options=[
                {'label': 'Duck', 'value': 'Duck'},
                {'label': 'Chicken', 'value': 'Chicken'},
                {'label': 'Turkey', 'value': 'Turkey'},
                {'label': 'Quail', 'value': 'Quail'},
                {'label': 'Goose', 'value': 'Goose'},
                {'label': 'Guinea fowl', 'value': 'Guinea fowl'},
                {'label': 'Pigeon', 'value': 'Pigeon'},
                {'label': 'Emu', 'value': 'Emu'},
                {'label': 'Partridge', 'value': 'Partridge'},
                {'label': 'Ostrich', 'value': 'Ostrich'},
                {'label': 'Cornish hen', 'value': 'Cornish hen'},
                {'label': 'Pheasant', 'value': 'Pheasant'},
                {'label': 'Squab', 'value': 'Squab'},
                {'label': 'Quinea pig', 'value': 'Quinea pig'},
                {'label': 'Capon', 'value': 'Capon'},
                {'label': 'Muscovy duck', 'value': 'Muscovy duck'},
                {'label': 'Guinea hen', 'value': 'Guinea hen'},
                {'label': 'Grouse', 'value': 'Grouse'},
                {'label': 'Teal', 'value': 'Teal'},
                {'label': 'Woodcock', 'value': 'Woodcock'},
                {'label': 'Wild turkey', 'value': 'Wild turkey'},
                {'label': 'Duck eggs', 'value': 'Duck eggs'},
                {'label': 'Chicken eggs', 'value': 'Chicken eggs'},
                {'label': 'Goose eggs', 'value': 'Goose eggs'},
                {'label': 'King banana', 'value': 'King banana'},
                {'label': 'Large fresh shrimp', 'value': 'Large fresh shrimp'},
                {'label': 'Mackerel fish', 'value': 'Mackerel fish'},
                {'label': 'Sunu fish', 'value': 'Sunu fish'},
                {'label': 'Fresh chives', 'value': 'Fresh chives'},
                {'label': 'Cashew apple', 'value': 'Cashew apple'},
                {'label': 'Negri fruit', 'value': 'Negri fruit'},
                {'label': 'Fresh Menteng', 'value': 'Fresh Menteng'},
                {'label': 'Steamed rice', 'value': 'Steamed rice'},
                {'label': 'Bentul (komba)', 'value': 'Bentul (komba)'},
                {'label': 'Soursop', 'value': 'Soursop'},
                {'label': 'Purslane leaves', 'value': 'Purslane leaves'},
                {'label': 'Pandanus leaf', 'value': 'Pandanus leaf'},
                {'label': 'Bali gunda leaf', 'value': 'Bali gunda leaf'},
                {'label': 'White sweet potato', 'value': 'White sweet potato'},
                {'label': 'Java plum', 'value': 'Java plum'},
                {'label': 'Sardines', 'value': 'Sardines'},
                {'label': 'Yellow sweet potato', 'value': 'Yellow sweet potato'},
                {'label': 'Fresh soybeans', 'value': 'Fresh soybeans'},
                {'label': 'Fresh potatoes', 'value': 'Fresh potatoes'},
                {'label': 'Dried peanuts', 'value': 'Dried peanuts'},
                {'label': 'Fresh green beans', 'value': 'Fresh green beans'},
                {'label': 'Papaya leaves', 'value': 'Papaya leaves'},
                {'label': 'Genjer', 'value': 'Genjer'},
                {'label': 'Fresh spinach', 'value': 'Fresh spinach'},
                {'label': 'Fresh dogfruit', 'value': 'Fresh dogfruit'},
                {'label': 'Salted duck eggs', 'value': 'Salted duck eggs'},
                {'label': 'Cheese', 'value': 'Cheese'},
                {'label': 'Young breadfruit', 'value': 'Young breadfruit'},
                {'label': 'Moringa leaves', 'value': 'Moringa leaves'},
                {'label': 'Black potatoes', 'value': 'Black potatoes'},
                {'label': 'Arrowroot', 'value': 'Arrowroot'},
                {'label': 'Catfish', 'value': 'Catfish'},
                {'label': 'Andaliman', 'value': 'Andaliman'},
                {'label': 'Peeled corn', 'value': 'Peeled corn'},
                {'label': 'Peanut gude', 'value': 'Peanut gude'},
                {'label': 'Bakung', 'value': 'Bakung'},
                {'label': 'Yam', 'value': 'Yam'},
                {'label': 'Mushroom', 'value': 'Mushroom'},
                {'label': 'Baligo', 'value': 'Baligo'},
                {'label': 'Matel ambon leaves', 'value': 'Matel ambon leaves'},
                {'label': 'Kacang komak', 'value': 'Kacang komak'},
                {'label': 'Carica papaya', 'value': 'Carica papaya'},
                {'label': 'Wild leaves', 'value': 'Wild leaves'},
                {'label': 'Lebui beans', 'value': 'Lebui beans'},
                {'label': 'Sago mushrooms', 'value': 'Sago mushrooms'},
                {'label': 'Langsat', 'value': 'Langsat'},
                {'label': 'Red beans', 'value': 'Red beans'},
                {'label': 'Sweet potato', 'value': 'Sweet potato'},
                {'label': 'Oyster mushrooms', 'value': 'Oyster mushrooms'},
                {'label': 'Bangun-bangun leaves', 'value': 'Bangun-bangun leaves'},
                {'label': 'Jatropha', 'value': 'Jatropha'},
                {'label': 'Kool flowers', 'value': 'Kool flowers'},
                {'label': 'Gandaria leaves', 'value': 'Gandaria leaves'},
                {'label': 'Uci beans', 'value': 'Uci beans'},
                {'label': 'Tarmon fish', 'value': 'Tarmon fish'},
                {'label': 'Titang fish', 'value': 'Titang fish'},
                {'label': 'Steamed rice', 'value': 'Steamed rice'},
                {'label': 'Mackerel fish', 'value': 'Mackerel fish'},
                {'label': 'Sardine', 'value': 'Sardine'},
                {'label': 'Pangium leaf', 'value': 'Pangium leaf'},
                {'label': 'Rukam fruit', 'value': 'Rukam fruit'},
                {'label': 'Freshwater prawn', 'value': 'Freshwater prawn'},
                {'label': 'Fern leaves', 'value': 'Fern leaves'},
                {'label': 'Java plum', 'value': 'Java plum'},
                {'label': 'Soursop', 'value': 'Soursop'},
                {'label': 'Purslane leaves', 'value': 'Purslane leaves'},
                {'label': 'Pandanus leaf', 'value': 'Pandanus leaf'},
                {'label': 'Bali gunda leaf', 'value': 'Bali gunda leaf'},
                {'label': 'Matoa', 'value': 'Matoa'},
                {'label': 'White sweet potato', 'value': 'White sweet potato'},
                {'label': 'Kemang leaves', 'value': 'Kemang leaves'},
                {'label': 'Kenikir leaves', 'value': 'Kenikir leaves'},
            ], multi=True, placeholder='Select favorite poultry'),

            html.Label("Please select your choice of grains and legumes:"),
            dcc.Dropdown(id='grains-legumes-dropdown', options=[
                {'label': 'Milled rice', 'value': 'Milled rice'},
                {'label': 'Black glutinous rice', 'value': 'Black glutinous rice'},
                {'label': 'Red rice', 'value': 'Red rice'},
                {'label': 'Seaweed', 'value': 'Seaweed'},
                {'label': 'Raw macaroni', 'value': 'Raw macaroni'},
                {'label': 'Tofu', 'value': 'Tofu'},
                {'label': 'Black rice', 'value': 'Black rice'},
                {'label': 'Boiled Red Beans', 'value': 'Boiled Red Beans'},
                {'label': 'Jali raw', 'value': 'Jali raw'},
                {'label': 'Bean sprouts', 'value': 'Bean sprouts'},
                {'label': 'Pohpohan leaves', 'value': 'Pohpohan leaves'},
                {'label': 'Lumai/lelunca', 'value': 'Lumai/lelunca'},
                {'label': 'Quail eggs', 'value': 'Quail eggs'},
                {'label': 'Red snapper', 'value': 'Red snapper'},
                {'label': 'Clam', 'value': 'Clam'},
                {'label': 'Duck eggs', 'value': 'Duck eggs'},
                {'label': 'Catfish', 'value': 'Catfish'},
                {'label': 'Sepat fish', 'value': 'Sepat fish'},
                {'label': 'Bluntas leaves', 'value': 'Bluntas leaves'},
                {'label': 'Talas Pontianak', 'value': 'Talas Pontianak'},
                {'label': 'Fresh kluwih', 'value': 'Fresh kluwih'},
                {'label': 'Kalaban fish', 'value': 'Kalaban fish'},
                {'label': 'Coconut shoots', 'value': 'Coconut shoots'},
                {'label': 'Lontar', 'value': 'Lontar'},
                {'label': 'Tunis beans', 'value': 'Tunis beans'},
                {'label': 'Lemuru fish', 'value': 'Lemuru fish'},
                {'label': 'White bitter melon', 'value': 'White bitter melon'},
                {'label': 'Terubuk sugarcane', 'value': 'Terubuk sugarcane'},
                {'label': 'Keribang', 'value': 'Keribang'},
                {'label': 'Gnetum gnemon', 'value': 'Gnetum gnemon'},
                {'label': 'Durian', 'value': 'Durian'},
                {'label': 'Lepok', 'value': 'Lepok'},
                {'label': 'Kawista', 'value': 'Kawista'},
                {'label': 'Soursop', 'value': 'Soursop'},
                {'label': 'Roasted peanuts', 'value': 'Roasted peanuts'},
                {'label': 'Matoa', 'value': 'Matoa'},
                {'label': 'Kincai beans', 'value': 'Kincai beans'},
                {'label': 'White sweet potatoes', 'value': 'White sweet potatoes'},
                {'label': 'Anchovies', 'value': 'Anchovies'},
                {'label': 'Bogor taro', 'value': 'Bogor taro'},
                {'label': 'Red saga peeled', 'value': 'Red saga peeled'},
                {'label': 'Gembili', 'value': 'Gembili'},
                {'label': 'Passion fruit', 'value': 'Passion fruit'},
                {'label': 'Melon', 'value': 'Melon'},
                {'label': 'Elephant foot yam', 'value': 'Elephant foot yam'},
            ], multi=True, placeholder='Select favorite grains and legumes'),
            
            html.Label("Please select your fruits and vegetables of choice:"),
            dcc.Dropdown(id='fruits-vegetables-dropdown', options=[
                {'label': 'Banana blossom', 'value': 'Banana blossom'},
                {'label': 'Apple', 'value': 'Apple'},
                {'label': 'Pineapple', 'value': 'Pineapple'},
                {'label': 'Avocado', 'value': 'Avocado'},
                {'label': 'Mangosteen', 'value': 'Mangosteen'},
                {'label': 'Bali oranges', 'value': 'Bali oranges'},
                {'label': 'Ruruhi fruit', 'value': 'Ruruhi fruit'},
                {'label': 'Guava', 'value': 'Guava'},
                {'label': 'Rambutan', 'value': 'Rambutan'},
                {'label': 'Matoa', 'value': 'Matoa'},
                {'label': 'Small persimmon', 'value': 'Small persimmon'},
                {'label': 'Java plum', 'value': 'Java plum'},
                {'label': 'Small sapodilla', 'value': 'Small sapodilla'},
                {'label': 'Star gooseberry leaf', 'value': 'Star gooseberry leaf'},
                {'label': 'Young corn', 'value': 'Young corn'},
                {'label': 'Yardlong bean', 'value': 'Yardlong bean'},
                {'label': 'Fresh cucumber', 'value': 'Fresh cucumber'},
                {'label': 'Fresh spinach', 'value': 'Fresh spinach'},
                {'label': 'Ear mushroom', 'value': 'Ear mushroom'},
                {'label': 'Water spinach', 'value': 'Water spinach'},
                {'label': 'Fresh radish', 'value': 'Fresh radish'},
                {'label': 'Fresh watercress', 'value': 'Fresh watercress'},
                {'label': 'Fresh eggplant', 'value': 'Fresh eggplant'},
                {'label': 'Fresh mustard greens', 'value': 'Fresh mustard greens'},
                {'label': 'Fresh carrots', 'value': 'Fresh carrots'},
                {'label': 'Parsley', 'value': 'Parsley'},
                {'label': 'Basil leaves', 'value': 'Basil leaves'},
                {'label': 'Lompong taro leaves', 'value': 'Lompong taro leaves'},
                {'label': 'Tekokak', 'value': 'Tekokak'},
                {'label': 'Caisin', 'value': 'Caisin'},
                {'label': 'Bitter bean', 'value': 'Bitter bean'},
                {'label': 'Gelang leaves', 'value': 'Gelang leaves'},
                {'label': 'Kwini mango', 'value': 'Kwini mango'},
                {'label': 'Jicama', 'value': 'Jicama'},
                {'label': 'Komak beans', 'value': 'Komak beans'},
                {'label': 'Gedi leaves', 'value': 'Gedi leaves'},
                {'label': 'Bali gunda leaves', 'value': 'Bali gunda leaves'},
                {'label': 'Bogor beans', 'value': 'Bogor beans'},
                {'label': 'Kemang leaves', 'value': 'Kemang leaves'},
                {'label': 'Kenikir leaves', 'value': 'Kenikir leaves'},
                {'label': 'Water chestnut, taro', 'value': 'Water chestnut, taro'},
                {'label': 'Keribang, yam', 'value': 'Keribang, yam'},
                {'label': 'Noni leaf', 'value': 'Noni leaf'},
                {'label': 'Chayote', 'value': 'Chayote'},
                {'label': 'Pandanus leaf', 'value': 'Pandanus leaf'},
                {'label': 'Ripe gandaria', 'value': 'Ripe gandaria'},
                {'label': 'Water chestnut taro', 'value': 'Water chestnut taro'},
                {'label': 'Anchovies', 'value': 'Anchovies'},
            ], multi=True, placeholder='Select favorite fruits and vegetables'),

        ], className="mb-5 p-3 border shadow"), width=6),

        # Button Section
        dbc.Col(html.Div([
            html.Button("Submit information", id="submit-button", className="btn btn-primary")
        ], className="text-center mt-3"), width=12),

    ]),

    # Day Information button
    dbc.Row([
        dbc.Col(dcc.Slider(id='day-slider', min=1, max=10, step=1, value=1, marks={i: f'Day {i}' for i in range(1, 11)}), width=6, className="mt-3", id="mt-3"),
        dbc.Col(html.Button("Next Day", id="update-button", className="btn btn-primary mt-3"), width=6, className="text-center")
    ]),
    
    # Display of results
    dbc.Row([
        dbc.Col(html.Div(id='results-area'), width=12)
    ]),

    # Image display sections added to the container
    dbc.Row([
    dbc.Col(html.Img(id='image_1', src='', style={'width': '100%', 'padding': '10px'}), width=6),
    dbc.Col(html.Img(id='image_2', src='', style={'width': '100%', 'padding': '10px'}), width=6)
])



], fluid=True)


from dash.exceptions import PreventUpdate
@app.callback(
    [Output('results-area', 'children'),
     Output('day-slider', 'value'),
     Output('image_1', 'src'),
     Output('image_2', 'src')],
    [Input('submit-button', 'n_clicks'),
     Input('update-button', 'n_clicks')],
    [dash.dependencies.State('sport-dropdown', 'value'),
     dash.dependencies.State('age-dropdown', 'value'),
     dash.dependencies.State('gender-dropdown', 'value'),
     dash.dependencies.State('day-slider', 'value'),
     dash.dependencies.State('poultry-dropdown', 'value'),
     dash.dependencies.State('grains-legumes-dropdown', 'value'),
     dash.dependencies.State('fruits-vegetables-dropdown', 'value')]
)
def update_output(submit_clicks, update_clicks, selected_sport, selected_age, selected_gender, selected_day,
                  selected_poultry, selected_grains, selected_vegetables):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'submit-button':
        selected_foods = selected_poultry + selected_grains + selected_vegetables
        best_solution = EA.run(athlete=selected_sport, gender=selected_gender, age_group=selected_age, food_choices=selected_foods)
        temp_day = 1
        for i in best_solution:
            EA.plotting_image(i, temp_day)
            temp_day+=1

        results_data = EA.chromosome_dictionary(selected_day)
        return generate_results_table(results_data, selected_day, selected_sport, selected_age, selected_gender), selected_day, \
            f'/assets/day_{selected_day}_barChart.png', \
            f'/assets/day_{selected_day}_pieChart.png'

    elif button_id == 'update-button':
        new_day = selected_day + 1 if selected_day < 10 else 1
        results_data = EA.chromosome_dictionary(new_day)
        return generate_results_table(results_data, new_day, selected_sport, selected_age, selected_gender), new_day, \
            f'/assets/day_{selected_day}_barChart.png', \
            f'/assets/day_{selected_day}_pieChart.png'

    else:
        raise PreventUpdate


def generate_results_table(results_data, day, selected_sport, selected_age, selected_gender):
    t_protein = EA.target_protein
    t_fat = EA.target_fat
    t_carb = EA.target_carbs
    t_calories = EA.calories

    target_table = html.Table([
        html.Thead(html.Tr([html.Th("Target Calories"), html.Th("Target Carbs"), html.Th("Target Proteins"), html.Th("Target Fats")])),
        html.Tbody([html.Tr([html.Td(t_calories), html.Td(t_carb), html.Td(t_protein), html.Td(t_fat)])])
    ], className="table table-striped")

    results_table = html.Table([
        html.Thead(html.Tr([html.Th("Food Item"), html.Th("Unit (g)"), html.Th("Protein (g)"), html.Th("Fat (g)"), html.Th("Carbohydrates (g)")])),
        html.Tbody([html.Tr([html.Td(food["food_item"]), html.Td(food["unit (g)"]), html.Td(food["protein"]), html.Td(food["fat"]), html.Td(food["carbohydrates"])])
                    for food in results_data])
    ], className="table table-responsive-sm table-hover")
    
    target_table_title = "Target Nutritional Values for: " + selected_gender + " " + selected_sport + ", Age: " + selected_age
    return html.Div([
        html.H4(f"Results for Day {day}", style={"margin-top": "20px", "text-align": "center"}),
        html.Hr(),
        html.H5(target_table_title, style={"text-align": "center"}),
        target_table,
        results_table
    ])


# Run the app
if __name__ == '__main__':
    url = "http://127.0.0.1:8050 "
    # Open the URL in a new browser window
    webbrowser.open_new(url)
    app.run_server(debug=False)
