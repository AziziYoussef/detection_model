import json
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_model_performance():
    """Analyse la performance de vos modèles et donne une évaluation contextuelle"""
    
    print("="*80)
    print("📊 ANALYSE DE LA PERFORMANCE DE VOS MODÈLES")
    print("="*80)
    
    # Charger vos résultats
    results_file = 'coco_evaluation/results/complete_evaluation_results.json'
    
    if not os.path.exists(results_file):
        print("❌ Fichiers de résultats non trouvés!")
        print("Lancez d'abord: python evaluate_all_models_complete.py")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Trouver le meilleur modèle
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]
    
    print(f"🏆 VOTRE MEILLEUR MODÈLE: {best_model_name}")
    print("="*60)
    print(f"📊 Précision (Precision): {best_model['precision']:.1%}")
    print(f"🔍 Rappel (Recall): {best_model['recall']:.1%}")
    print(f"⭐ F1-Score: {best_model['f1']:.1%}")
    print(f"💪 Confiance: {best_model['confidence']:.1%}")
    
    # Benchmarks de référence pour la détection d'objets
    benchmarks = {
        'Débutant': {'precision': 0.30, 'recall': 0.40, 'f1': 0.35},
        'Bon': {'precision': 0.50, 'recall': 0.60, 'f1': 0.55},
        'Très bon': {'precision': 0.70, 'recall': 0.75, 'f1': 0.72},
        'Excellent': {'precision': 0.85, 'recall': 0.85, 'f1': 0.85},
        'SOTA (State-of-the-art)': {'precision': 0.90, 'recall': 0.90, 'f1': 0.90}
    }
    
    print(f"\n📈 ÉVALUATION CONTEXTUELLE")
    print("="*60)
    
    # Analyser le F1-Score (métrique principale)
    f1_score = best_model['f1']
    
    if f1_score >= 0.85:
        level = "🏆 EXCELLENT - Niveau professionnel"
        color = "🟢"
    elif f1_score >= 0.72:
        level = "⭐ TRÈS BON - Très satisfaisant"
        color = "🟢"
    elif f1_score >= 0.55:
        level = "👍 BON - Satisfaisant pour la plupart des applications"
        color = "🟡"
    elif f1_score >= 0.35:
        level = "🔄 DÉBUTANT - Perfectible mais fonctionnel"
        color = "🟠"
    else:
        level = "❌ FAIBLE - Nécessite des améliorations"
        color = "🔴"
    
    print(f"{color} NIVEAU GLOBAL: {level}")
    print(f"📊 Votre F1-Score: {f1_score:.1%}")
    
    # Comparaison détaillée
    print(f"\n🎯 COMPARAISON AVEC LES STANDARDS")
    print("="*60)
    
    for benchmark_name, benchmark_values in benchmarks.items():
        status = "✅" if f1_score >= benchmark_values['f1'] else "❌"
        print(f"{status} {benchmark_name:<25} F1: {benchmark_values['f1']:.1%}")
    
    # Analyse spécifique à votre domaine
    print(f"\n🔍 ANALYSE SPÉCIFIQUE - DÉTECTION D'OBJETS PERDUS")
    print("="*60)
    
    precision = best_model['precision']
    recall = best_model['recall']
    
    # Analyse de la précision
    if precision >= 0.70:
        prec_eval = "🎯 Excellente - Peu de fausses alarmes"
    elif precision >= 0.50:
        prec_eval = "👍 Bonne - Acceptable pour un système réel"
    elif precision >= 0.30:
        prec_eval = "⚠️ Moyenne - Beaucoup de fausses détections"
    else:
        prec_eval = "❌ Faible - Trop de fausses alarmes"
    
    # Analyse du rappel
    if recall >= 0.80:
        rec_eval = "🔍 Excellent - Trouve presque tous les objets"
    elif recall >= 0.60:
        rec_eval = "👍 Bon - Trouve la plupart des objets"
    elif recall >= 0.40:
        rec_eval = "⚠️ Moyen - Manque certains objets"
    else:
        rec_eval = "❌ Faible - Manque beaucoup d'objets"
    
    print(f"Précision: {precision:.1%} → {prec_eval}")
    print(f"Rappel: {recall:.1%} → {rec_eval}")
    
    # Recommandations spécifiques
    print(f"\n💡 RECOMMANDATIONS")
    print("="*60)
    
    if precision < 0.50:
        print("🔧 Améliorer la précision:")
        print("   • Augmenter le seuil de confiance")
        print("   • Plus d'exemples négatifs dans l'entraînement")
        print("   • Fine-tuning avec des données plus propres")
    
    if recall < 0.60:
        print("🔧 Améliorer le rappel:")
        print("   • Diminuer le seuil de confiance")
        print("   • Augmentation de données (data augmentation)")
        print("   • Plus d'exemples positifs variés")
    
    if f1_score >= 0.45:
        print("✅ Votre modèle est utilisable en production!")
        print("   • Performance acceptable pour un système réel")
        print("   • Peut être déployé avec monitoring")
    
    # Comparaison avec modèles célèbres
    print(f"\n🌟 COMPARAISON AVEC MODÈLES CÉLÈBRES")
    print("="*60)
    
    famous_models = {
        "YOLO v3 (2018)": 0.51,
        "SSD MobileNet": 0.48,
        "Faster R-CNN": 0.55,
        "YOLO v5 (2020)": 0.65,
        "YOLO v8 (2023)": 0.75
    }
    
    your_rank = 1
    for model_name, model_f1 in famous_models.items():
        if f1_score < model_f1:
            your_rank += 1
        
        comparison = "🟢" if f1_score >= model_f1 else "🔴"
        print(f"{comparison} {model_name:<20} F1: {model_f1:.1%}")
    
    print(f"\n🏅 VOTRE CLASSEMENT: {your_rank}/{len(famous_models)+1}")
    
    if your_rank <= 3:
        print("🎉 Excellent! Vous rivalisez avec les meilleurs modèles!")
    elif your_rank <= 5:
        print("👍 Très bien! Performance comparable aux modèles standards")
    else:
        print("🔄 Perfectible, mais c'est un bon début!")
    
    # Visualisation
    create_performance_chart(best_model, benchmarks, famous_models)
    
    # Résumé final
    print(f"\n🎯 RÉSUMÉ FINAL")
    print("="*60)
    
    if f1_score >= 0.50:
        final_verdict = "🟢 MODÈLE PRÊT POUR LA PRODUCTION"
        advice = "Votre modèle a une performance suffisante pour être déployé dans un système réel d'objets perdus."
    elif f1_score >= 0.35:
        final_verdict = "🟡 MODÈLE FONCTIONNEL MAIS PERFECTIBLE"
        advice = "Votre modèle fonctionne mais pourrait bénéficier d'améliorations avant déploiement."
    else:
        final_verdict = "🔴 MODÈLE À AMÉLIORER"
        advice = "Le modèle nécessite plus d'entraînement ou d'optimisation avant utilisation."
    
    print(f"Verdict: {final_verdict}")
    print(f"Conseil: {advice}")
    
    return best_model, f1_score

def create_performance_chart(best_model, benchmarks, famous_models):
    """Crée un graphique de performance"""
    
    # Données pour le graphique
    categories = ['Précision', 'Rappel', 'F1-Score']
    your_scores = [
        best_model['precision'],
        best_model['recall'], 
        best_model['f1']
    ]
    
    good_benchmark = [
        benchmarks['Bon']['precision'],
        benchmarks['Bon']['recall'],
        benchmarks['Bon']['f1']
    ]
    
    excellent_benchmark = [
        benchmarks['Excellent']['precision'],
        benchmarks['Excellent']['recall'],
        benchmarks['Excellent']['f1']
    ]
    
    # Créer le graphique
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, your_scores, width, label='Votre modèle', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, good_benchmark, width, label='Niveau "Bon"', color='#A23B72', alpha=0.6)
    bars3 = ax.bar(x + width, excellent_benchmark, width, label='Niveau "Excellent"', color='#F18F01', alpha=0.6)
    
    # Personnalisation
    ax.set_ylabel('Score')
    ax.set_title('Comparaison de Performance de Votre Modèle')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs('performance_analysis', exist_ok=True)
    plt.savefig('performance_analysis/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Graphique sauvegardé: performance_analysis/model_performance_comparison.png")

def main():
    best_model, f1_score = analyze_model_performance()
    
    print(f"\n💬 EN RÉSUMÉ:")
    print("="*40)
    
    if f1_score >= 0.50:
        print("🎉 Félicitations! Votre modèle a une BONNE performance!")
        print("   Vous pouvez être fier de ce résultat.")
    elif f1_score >= 0.35:
        print("👍 Votre modèle a une performance CORRECTE!")
        print("   C'est un bon début, avec du potentiel d'amélioration.")
    else:
        print("🔄 Votre modèle a besoin d'améliorations.")
        print("   Mais c'est normal pour un premier modèle!")
    
    print(f"\n🎯 Pour la détection d'objets perdus, un F1-Score de {f1_score:.1%} est:")
    
    if f1_score >= 0.45:
        print("✅ SUFFISANT pour un déploiement en conditions réelles")
    elif f1_score >= 0.30:
        print("⚠️ UTILISABLE avec supervision humaine")
    else:
        print("❌ INSUFFISANT pour une utilisation pratique")

if __name__ == "__main__":
    main()