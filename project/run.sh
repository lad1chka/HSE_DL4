#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Мультимодальная система прогнозирования${NC}"
echo -e "${BLUE}       волатильности акций и ETF${NC}"
echo -e "${BLUE}========================================${NC}\n"

show_menu() {
    echo -e "${YELLOW}Выберите действие:${NC}"
    echo "1) Загрузить и подготовить данные (data_preparation.py)"
    echo "2) Создать признаки (feature_engineering.py)"
    echo "3) Обучить модель (model_training.py)"
    echo "4) Запустить интерфейс Gradio (app.py)"
    echo "5) Запустить весь пайплайн (1→2→3→4)"
    echo "6) Выход"
    echo ""
}

while true; do
    show_menu
    read -p "Введите номер (1-6): " choice
    
    case $choice in
        1)
            echo -e "\n${GREEN}Запуск подготовки данных...${NC}\n"
            python3 data_preparation.py
            echo ""
            ;;
        2)
            echo -e "\n${GREEN}Запуск создания признаков...${NC}\n"
            python3 feature_engineering.py
            echo ""
            ;;
        3)
            echo -e "\n${GREEN}Запуск обучения модели...${NC}\n"
            python3 model_training.py
            echo ""
            ;;
        4)
            echo -e "\n${GREEN}Запуск интерфейса Gradio...${NC}\n"
            echo -e "${YELLOW}Интерфейс будет доступен по адресу: http://127.0.0.1:7860${NC}\n"
            python3 app.py
            ;;
        5)
            echo -e "\n${GREEN}Запуск полного пайплайна...${NC}\n"
            
            echo -e "${BLUE}Шаг 1: Подготовка данных${NC}"
            python3 data_preparation.py
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}⚠ Предупреждение: data_preparation.py завершилась с ошибкой${NC}"
                echo -e "${YELLOW}Убедитесь, что репозиторий stocknet-dataset клонирован.${NC}"
            fi
            
            echo -e "\n${BLUE}Шаг 2: Создание признаков${NC}"
            python3 feature_engineering.py
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}❌ Ошибка на шаге 2${NC}"
                break
            fi
            
            echo -e "\n${BLUE}Шаг 3: Обучение модели${NC}"
            python3 model_training.py
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}❌ Ошибка на шаге 3${NC}"
                break
            fi
            
            echo -e "\n${BLUE}Шаг 4: Запуск интерфейса${NC}"
            echo -e "${YELLOW}Интерфейс будет доступен по адресу: http://127.0.0.1:7860${NC}\n"
            python3 app.py
            ;;
        6)
            echo -e "\n${GREEN}До встречи!${NC}\n"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Неверный выбор. Попробуйте снова.${NC}\n"
            ;;
    esac
done
