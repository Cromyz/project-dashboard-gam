from googleads import ad_manager

def run_diagnostic():
    print("🔄 Initialisation du client GAM...")
    client = ad_manager.AdManagerClient.LoadFromStorage('googleads.yaml')
    API_VERSION = 'v202602'
    
    line_item_service = client.GetService('LineItemService', version=API_VERSION)
    
    print("\n🔍 TEST 3 : Le compte voit-il AU MOINS UNE campagne sur tout le réseau ?")
    
    # On demande juste 1 seul résultat, peu importe lequel
    statement_any = ad_manager.StatementBuilder(version=API_VERSION).Limit(1)
    
    try:
        response = line_item_service.getLineItemsByStatement(statement_any.ToStatement())
        if 'results' in response and len(response['results']) > 0:
            print(f"✅ SUCCÈS : Le compte n'est pas aveugle !")
            print(f"   Il a réussi à voir la campagne ID : {response['results'][0]['id']} (Nom : {response['results'][0]['name']})")
            print("   👉 CONCLUSION : Le compte de service marche bien, mais il n'a pas été ajouté à la bonne 'Équipe' pour voir votre campagne ID 7206464132.")
        else:
            print("❌ ÉCHEC TOTAL : L'API répond 0 résultat.")
            print("   👉 CONCLUSION : Le compte de service est complètement aveugle. L'administrateur a oublié de cocher la case 'Accès à l'API' dans le Rôle de l'utilisateur, ou ne l'a pas mis dans l'équipe 'Toutes les entités'.")
    except Exception as e:
        print(f"⚠️ ERREUR : {e}")

if __name__ == '__main__':
    run_diagnostic()