"""
Визуализация результатов классификации PDF.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Настройка для корректного отображения
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 10)


def load_report(path: str = "results_report.json") -> dict:
    """Загружает отчёт из JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_visualizations(report: dict, output_dir: str = "."):
    """Создаёт графики и таблицы."""
    summary = report['summary']
    results = report['results']

    # Создаём фигуру с subplot'ами
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Результаты классификации PDF файлов', fontsize=16, fontweight='bold')

    # 1. Общая статистика (верхний левый)
    ax1 = fig.add_subplot(2, 3, 1)
    statuses = ['Успешно', 'В мусор', 'Ошибки']
    values = [summary['successful'], summary['moved_to_trash'], summary['errors']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bars = ax1.bar(statuses, values, color=colors, edgecolor='black')
    ax1.set_title('Статус обработки', fontweight='bold')
    ax1.set_ylabel('Количество файлов')

    # Добавляем значения на столбцы
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')

    ax1.set_ylim(0, max(values) * 1.15)

    # 2. Распределение по странам (верхний центр)
    ax2 = fig.add_subplot(2, 3, 2)
    countries = summary['by_country']
    country_names = list(countries.keys())
    country_values = list(countries.values())

    # Сортируем по убыванию
    sorted_idx = sorted(range(len(country_values)), key=lambda i: country_values[i], reverse=True)
    country_names = [country_names[i] for i in sorted_idx]
    country_values = [country_values[i] for i in sorted_idx]

    colors_country = plt.cm.Set3(range(len(country_names)))
    wedges, texts, autotexts = ax2.pie(
        country_values,
        labels=country_names,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(country_values))})',
        colors=colors_country,
        explode=[0.05 if i == 0 else 0 for i in range(len(country_names))],
        shadow=True
    )
    ax2.set_title('Распределение по странам', fontweight='bold')

    # 3. Распределение по типам документов (верхний правый)
    ax3 = fig.add_subplot(2, 3, 3)
    doc_types = summary['by_doc_type']
    type_names = list(doc_types.keys())
    type_values = list(doc_types.values())

    # Сортируем по убыванию
    sorted_idx = sorted(range(len(type_values)), key=lambda i: type_values[i], reverse=True)
    type_names = [type_names[i] for i in sorted_idx]
    type_values = [type_values[i] for i in sorted_idx]

    colors_type = plt.cm.Paired(range(len(type_names)))
    wedges, texts, autotexts = ax3.pie(
        type_values,
        labels=type_names,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(type_values))})',
        colors=colors_type,
        explode=[0.05 if i == 0 else 0 for i in range(len(type_names))],
        shadow=True
    )
    ax3.set_title('Распределение по типам документов', fontweight='bold')

    # 4. Горизонтальная гистограмма типов (нижний левый)
    ax4 = fig.add_subplot(2, 3, 4)
    y_pos = range(len(type_names))
    ax4.barh(y_pos, type_values, color=colors_type, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(type_names)
    ax4.set_xlabel('Количество файлов')
    ax4.set_title('Типы документов (детально)', fontweight='bold')

    # Добавляем значения
    for i, v in enumerate(type_values):
        ax4.text(v + 3, i, str(v), va='center', fontweight='bold')
    ax4.set_xlim(0, max(type_values) * 1.15)

    # 5. Таблица статистики (нижний центр)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    # Собираем статистику по компаниям
    companies = {}
    for r in results:
        if r['classification'] and r['classification']['company']:
            company = r['classification']['company']
            companies[company] = companies.get(company, 0) + 1

    # Топ-10 компаний
    top_companies = sorted(companies.items(), key=lambda x: -x[1])[:10]

    table_data = [
        ['Метрика', 'Значение'],
        ['Всего файлов', str(summary['total_files'])],
        ['Успешно классифицировано', f"{summary['successful']} ({summary['successful']/summary['total_files']*100:.1f}%)"],
        ['В мусор (низкий confidence)', str(summary['moved_to_trash'])],
        ['Ошибки', str(summary['errors'])],
        ['', ''],
        ['Уникальных стран', str(len(countries))],
        ['Уникальных типов', str(len(doc_types))],
        ['Уникальных компаний', str(len(companies))],
    ]

    table = ax5.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Стилизация заголовка таблицы
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax5.set_title('Сводная статистика', fontweight='bold', pad=20)

    # 6. Топ-10 компаний (нижний правый)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    company_table_data = [['Компания', 'Файлов']] + [
        [name[:40] + '...' if len(name) > 40 else name, str(count)]
        for name, count in top_companies
    ]

    table2 = ax6.table(
        cellText=company_table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.75, 0.25]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.4)

    # Стилизация заголовка
    for i in range(2):
        table2[(0, i)].set_facecolor('#3498db')
        table2[(0, i)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Топ-10 компаний', fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Сохраняем
    output_path = Path(output_dir) / 'classification_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Графики сохранены: {output_path}")

    plt.close()

    return output_path


def print_summary_table(report: dict):
    """Выводит текстовую таблицу в консоль."""
    summary = report['summary']
    results = report['results']

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ PDF")
    print("="*70)

    print(f"\n{'ОБЩАЯ СТАТИСТИКА':^70}")
    print("-"*70)
    print(f"{'Всего файлов:':<30} {summary['total_files']:>10}")
    print(f"{'Успешно:':<30} {summary['successful']:>10} ({summary['successful']/summary['total_files']*100:.1f}%)")
    print(f"{'В мусор:':<30} {summary['moved_to_trash']:>10}")
    print(f"{'Ошибки:':<30} {summary['errors']:>10}")

    print(f"\n{'РАСПРЕДЕЛЕНИЕ ПО СТРАНАМ':^70}")
    print("-"*70)
    for country, count in sorted(summary['by_country'].items(), key=lambda x: -x[1]):
        pct = count / summary['successful'] * 100
        bar = '#' * int(pct / 2)
        print(f"{country:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"\n{'РАСПРЕДЕЛЕНИЕ ПО ТИПАМ ДОКУМЕНТОВ':^70}")
    print("-"*70)
    for doc_type, count in sorted(summary['by_doc_type'].items(), key=lambda x: -x[1]):
        pct = count / summary['successful'] * 100
        bar = '#' * int(pct / 2)
        print(f"{doc_type:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    # Топ компаний
    companies = {}
    for r in results:
        if r['classification'] and r['classification']['company']:
            company = r['classification']['company']
            companies[company] = companies.get(company, 0) + 1

    print(f"\n{'ТОП-10 КОМПАНИЙ':^70}")
    print("-"*70)
    for company, count in sorted(companies.items(), key=lambda x: -x[1])[:10]:
        name = company[:45] + '...' if len(company) > 45 else company
        print(f"{name:<50} {count:>5}")

    # Ошибки
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        print(f"\n{'ФАЙЛЫ С ОШИБКАМИ':^70}")
        print("-"*70)
        for r in errors:
            print(f"  {Path(r['source']).name}")
            print(f"    Ошибка: {r['error']}")

    print("\n" + "="*70)


if __name__ == '__main__':
    report = load_report()
    print_summary_table(report)
    create_visualizations(report)
