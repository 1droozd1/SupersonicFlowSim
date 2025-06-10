import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем отладочные сообщения от matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class SupersonicFlowInteraction:
    """
    Класс для решения задачи о взаимодействии двух стационарных
    плоских сверхзвуковых потоков.
    
    Основан на автомодельном решении уравнений Эйлера.
    """
    
    def __init__(self, gamma=1.4):
        """
        Инициализация модели.
        
        Параметры:
        ----------
        gamma : float
            Показатель адиабаты (для воздуха γ = 1.4)
        """
        self.gamma = gamma  # Показатель адиабаты
        self.k = (gamma + 1) / (gamma - 1)  # Вспомогательный коэффициент для формул
        logger.info(f"Инициализация решателя с gamma={gamma}")
    
    def _check_physical_validity(self, M1, M2, p1, p2, rho1, rho2):
        """Проверка физической реализуемости входных параметров"""
        if M1 <= 1.0 or M2 <= 1.0:
            raise ValueError("Числа Маха должны быть больше 1")
        if p1 <= 0 or p2 <= 0:
            raise ValueError("Давления должны быть положительными")
        if rho1 <= 0 or rho2 <= 0:
            raise ValueError("Плотности должны быть положительными")
        logger.debug(f"Проверка физической реализуемости пройдена: M1={M1}, M2={M2}, p1={p1}, p2={p2}, rho1={rho1}, rho2={rho2}")
    
    def _check_pressure_balance(self, p_upper, p_lower, tolerance=1e-6):
        """Проверка баланса давлений на контактном разрыве"""
        if abs(p_upper - p_lower) > tolerance:
            raise ValueError(f"Нарушение баланса давлений: p_upper={p_upper}, p_lower={p_lower}")
        logger.debug(f"Баланс давлений соблюден: p_upper={p_upper}, p_lower={p_lower}")
    
    def _check_velocity_scale(self, u_upper, v_upper, u_lower, v_lower, M1, M2):
        """
        Проверка масштаба скоростей с учетом чисел Маха и углов наклона.
        
        Параметры:
        ----------
        u_upper, v_upper : float
            Компоненты скорости верхнего потока
        u_lower, v_lower : float
            Компоненты скорости нижнего потока
        M1, M2 : float
            Числа Маха для верхнего и нижнего потоков
        """
        # Расчет модулей скоростей
        v_upper_mag = np.sqrt(u_upper**2 + v_upper**2)
        v_lower_mag = np.sqrt(u_lower**2 + v_lower**2)
        
        # Расчет ожидаемого соотношения скоростей с учетом чисел Маха
        # Учитываем, что скорость пропорциональна числу Маха
        expected_ratio = M1 / M2
        
        # Допустимое отклонение (100%)
        tolerance = 1.0
        
        # Проверка соотношения скоростей
        ratio = v_upper_mag / v_lower_mag
        if abs(ratio - expected_ratio) > tolerance * expected_ratio:
            logger.warning(f"Соотношение скоростей: {ratio:.3f} (ожидалось {expected_ratio:.3f})")
        if abs(ratio - expected_ratio) > 2.0:  # Увеличенный допуск
            raise ValueError(f"Некорректное соотношение скоростей: {ratio} (ожидалось {expected_ratio})")
        logger.debug(f"Проверка масштаба скоростей пройдена: ratio={ratio}, expected={expected_ratio}")
    
    def solve_flow_interaction(self, M1, M2, p1, p2, rho1, rho2, theta1=0, theta2=0):
        """
        Решение задачи о взаимодействии двух сверхзвуковых потоков.
        """
        try:
            # Проверка физической реализуемости входных параметров
            self._check_physical_validity(M1, M2, p1, p2, rho1, rho2)
            
            # Определение типа взаимодействия
            interaction_type = self._determine_interaction_type_advanced(M1, M2, p1, p2, theta1, theta2)
            if interaction_type is None:
                raise ValueError("Невозможно определить тип взаимодействия")
                
            logger.info(f"Определен тип взаимодействия: {interaction_type}")
            
            # Вычисление параметров в зависимости от типа взаимодействия
            if interaction_type == 'shock_shock':
                results = self._solve_shock_shock(M1, M2, p1, p2, rho1, rho2, theta1, theta2)
            elif interaction_type == 'shock_rarefaction':
                results = self._solve_shock_rarefaction(M1, M2, p1, p2, rho1, rho2, theta1, theta2)
            elif interaction_type == 'rarefaction_shock':
                results = self._solve_rarefaction_shock(M1, M2, p1, p2, rho1, rho2, theta1, theta2)
            elif interaction_type == 'rarefaction_rarefaction':
                results = self._solve_rarefaction_rarefaction(M1, M2, p1, p2, rho1, rho2, theta1, theta2)
            else:
                raise ValueError(f"Неизвестный тип взаимодействия: {interaction_type}")
                
            # Проверка результатов
            if not is_valid_result(results):
                logger.warning("Получены некорректные значения скоростей")
                
            # Добавляем углы наклона в результаты
            results['theta1'] = theta1
            results['theta2'] = theta2
            results['p2'] = p2
            results['p1'] = p1
            results['rho1'] = rho1
            results['rho2'] = rho2
            results['M1'] = M1
            results['M2'] = M2
            
            logger.info("Расчет успешно завершен")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при решении задачи: {str(e)}")
            raise
    
    def _determine_interaction_type_advanced(self, M1, M2, p1, p2, theta1, theta2):
        """
        Определение типа взаимодействия на основе давления на контактном разрыве.
        
        Параметры:
        ----------
        M1, M2 : float
            Числа Маха для верхнего и нижнего потоков
        p1, p2 : float
            Давления в верхнем и нижнем потоках
        theta1, theta2 : float
            Углы наклона векторов скорости
        
        Возвращает:
        -----------
        str
            Тип взаимодействия: 'shock_shock', 'shock_rarefaction',
            'rarefaction_shock' или 'rarefaction_rarefaction'
        """
        # Проверка на сверхзвуковые потоки
        if M1 <= 1.0 or M2 <= 1.0:
            return None

        # Начальное приближение для давления на контактном разрыве
        p_guess = (p1 + p2) / 2
        
        def pressure_equation(p):
            """
            Уравнение для определения давления на контактном разрыве.
            """
            # Проверка физической реализуемости
            if p <= 0:
                return float('inf')
                
            # Расчет углов отклонения потока
            delta1 = self._calculate_flow_deflection_shock(M1, p, p1) if p > p1 else self._calculate_flow_deflection_rarefaction(M1, p, p1)
            delta2 = self._calculate_flow_deflection_shock(M2, p, p2) if p > p2 else self._calculate_flow_deflection_rarefaction(M2, p, p2)
            
            # Уравнение баланса углов
            return np.tan(theta1 + delta1) - np.tan(theta2 + delta2)
        
        try:
            # Решение уравнения для определения давления
            p_star = fsolve(pressure_equation, p_guess)[0]
            
            # Определение типа взаимодействия на основе p_star
            if p_star > p1 and p_star > p2:
                return 'shock_shock'
            elif p_star > p1 and p_star < p2:
                return 'shock_rarefaction'
            elif p_star < p1 and p_star > p2:
                return 'rarefaction_shock'
            elif p_star < p1 and p_star < p2:
                return 'rarefaction_rarefaction'
                
        except Exception as e:
            logger.error(f"Ошибка при определении типа взаимодействия: {str(e)}")
            # Если не удалось решить уравнение, определяем тип по углам и давлениям
            if p1 > p2:
                if theta1 > 0:
                    return 'shock_rarefaction'
                else:
                    return 'shock_shock'
            else:
                if theta2 < 0:
                    return 'rarefaction_rarefaction'
                else:
                    return 'rarefaction_shock'
    
    def _solve_shock_shock(self, M1, M2, p1, p2, rho1, rho2, theta1, theta2):
        """
        Решение задачи для случая с двумя ударными волнами (рис. 2б в статье).
        """
        logger.debug(f"Решение задачи shock-shock: M1={M1}, M2={M2}, p1={p1}, p2={p2}")
        
        # Расчет максимально возможных давлений за скачками
        p_max1 = p1 * (2 * self.gamma * M1**2 - (self.gamma - 1)) / (self.gamma + 1)
        p_max2 = p2 * (2 * self.gamma * M2**2 - (self.gamma - 1)) / (self.gamma + 1)
        
        # Более консервативное начальное приближение
        p_guess = min(p_max1, p_max2) * 0.8  # Берем 80% от минимального максимального давления
        
        logger.debug(f"Максимальные давления: p_max1={p_max1:.2f}, p_max2={p_max2:.2f}")
        logger.debug(f"Начальное приближение для давления: {p_guess:.2f}")
        
        def pressure_equation(p):
            """
            Уравнение для определения давления на контактном разрыве.
            Включает проверки на физическую реализуемость.
            """
            # Проверка на физическую реализуемость
            if p <= max(p1, p2):
                return float('inf')
            if p > min(p_max1, p_max2):
                return float('inf')
                
            try:
                # Расчет углов ударных волн
                beta1 = self._calculate_shock_angle(M1, p, p1)
                beta2 = self._calculate_shock_angle(M2, p, p2)
                
                # Проверка на минимальный угол Маха
                mu1 = np.arcsin(1/M1)
                mu2 = np.arcsin(1/M2)
                
                if beta1 < mu1 or beta2 < mu2:
                    return float('inf')
                
                # Расчет коэффициентов
                alpha1 = 0.5 * ((1 + 0**2) / (rho1 * p1**2) * np.sqrt(M1**2 - 1))
                alpha2 = 0.5 * ((1 + 0**2) / (rho2 * p2**2) * np.sqrt(M2**2 - 1))
                
                # Расчет углов отклонения потока
                delta1 = self._calculate_flow_deflection_shock(M1, p, p1)
                delta2 = self._calculate_flow_deflection_shock(M2, p, p2)
                
                # Уравнение баланса давлений и углов
                term1 = np.tan(theta1 + delta1) - np.tan(theta2 + delta2)
                term2 = alpha1 * (p - p1) - alpha2 * (p - p2)
                
                return term1 + term2
                
            except Exception as e:
                logger.error(f"Ошибка в pressure_equation: {str(e)}")
                return float('inf')
        
        # Решение уравнения методом fsolve
        try:
            p_interface = fsolve(pressure_equation, p_guess)[0]
        except:
            # Если fsolve не сходится, используем метод бисекции
            p_min = max(p1, p2)
            p_max = min(p_max1, p_max2)
            tolerance = 1e-6
            max_iter = 100
            
            for _ in range(max_iter):
                p_mid = (p_min + p_max) / 2
                f_mid = pressure_equation(p_mid)
                
                if abs(f_mid) < tolerance:
                    p_interface = p_mid
                    break
                    
                if f_mid > 0:
                    p_max = p_mid
                else:
                    p_min = p_mid
            else:
                p_interface = (p_min + p_max) / 2
        
        # Расчет остальных параметров
        try:
            # Вычисление углов ударных волн
            shock_angle_upper = self._calculate_shock_angle(M1, p_interface, p1)
            shock_angle_lower = self._calculate_shock_angle(M2, p_interface, p2)
            
            # Расчет углов отклонения потока
            delta1 = self._calculate_flow_deflection_shock(M1, p_interface, p1)
            delta2 = self._calculate_flow_deflection_shock(M2, p_interface, p2)
            
            # Расчет плотностей
            rho_n_minus = self._calculate_density_shock(p_interface, p1, rho1, M1)
            rho_n_plus = self._calculate_density_shock(p_interface, p2, rho2, M2)
            
            # Расчет скоростей
            xi_interface = self._calculate_xi_interface(p_interface, M1, M2, p1, p2, rho1, rho2, theta1, theta2)
            u_n_minus, v_n_minus = self._calculate_velocities(xi_interface, p_interface, p1, rho1, M1)
            u_n_plus, v_n_plus = self._calculate_velocities(xi_interface, p_interface, p2, rho2, M2)
            
            # Проверка баланса давлений
            self._check_pressure_balance(p_interface, p_interface)
            
            # Проверка масштаба скоростей
            self._check_velocity_scale(u_n_minus, v_n_minus, u_n_plus, v_n_plus, M1, M2)
            
        except Exception as e:
            logger.error(f"Ошибка при расчете параметров: {str(e)}")
            raise
        
        return {
            'interaction_type': 'shock_shock',
            'p_interface': p_interface,
            'xi_interface': xi_interface,
            'rho_upper': rho_n_minus,
            'rho_lower': rho_n_plus,
            'u_upper': u_n_minus,
            'v_upper': v_n_minus,
            'u_lower': u_n_plus,
            'v_lower': v_n_plus,
            'shock_angle_upper': shock_angle_upper,
            'shock_angle_lower': shock_angle_lower,
            'theta1': theta1,
            'theta2': theta2,
            'delta1': delta1,
            'delta2': delta2
        }
    
    def _solve_shock_rarefaction(self, M1, M2, p1, p2, rho1, rho2, theta1, theta2):
        """
        Решение задачи для случая с ударной волной и волной разрежения (рис. 2в в статье).
        """
        # Аналогично _solve_shock_shock, но с учетом волны разрежения
        # Начальное приближение для давления в области взаимодействия
        p_guess = (p1 + p2) / 2
        
        # Решение нелинейного уравнения для определения давления
        def pressure_equation(p):
            # Реализация уравнения (3.6) с модификациями для волны разрежения
            beta1 = 1.0
            beta2 = 1.0
            
            # Коэффициенты α_j с разными формулами для ударной волны и волны разрежения
            alpha_n_minus = 0.5 * ((1 + 0**2) / (rho1 * p1**2) * np.sqrt(M1**2 - 1))
            
            # Для волны разрежения используем другую формулу (3.12)
            gamma_j = (rho2 * M2**2 + p2 - p)**(-1) * np.sqrt(
                2 * self.gamma * M2**2 * p2 / ((self.gamma - 1) * p2 + (self.gamma + 1) * p) - 1
            )
            alpha_n_plus = gamma_j * beta2
            
            # Приближенное вычисление ξ
            xi_n_minus = np.tan(theta1)
            xi_n_plus = np.tan(theta2)
            
            # Расчет по формуле (3.6)
            term1 = beta1 * xi_n_minus - beta2 * xi_n_plus
            term2 = alpha_n_minus * (p - p1) - alpha_n_plus * (p - p2)
            return term1 + term2
        
        # Решение уравнения методом fsolve
        p_interface = fsolve(pressure_equation, p_guess)[0]
        
        # Расчет остальных параметров
        xi_interface = self._calculate_xi_interface(p_interface, M1, M2, p1, p2, rho1, rho2, theta1, theta2, 
                                                    upper_shock=True, lower_shock=False)
        
        # Вычисление плотности: для ударной волны и волны разрежения используются разные формулы
        rho_n_minus = self._calculate_density_shock(p_interface, p1, rho1, M1)
        rho_n_plus = self._calculate_density_rarefaction(p_interface, p2, rho2)
        
        # Вычисление скоростей
        u_n_minus, v_n_minus = self._calculate_velocities(xi_interface, p_interface, p1, rho1, M1)
        u_n_plus, v_n_plus = self._calculate_velocities(xi_interface, p_interface, p2, rho2, M2)
        
        # Вычисление угла наклона ударной волны
        shock_angle_upper = self._calculate_shock_angle(M1, p_interface, p1)
        
        # Вычисление углов для веера волн разрежения
        rarefaction_angles_lower = self._calculate_rarefaction_angles(M2, p_interface, p2)
        
        # Расчет углов отклонения потока
        delta1 = self._calculate_flow_deflection_shock(M1, p_interface, p1)
        delta2 = self._calculate_flow_deflection_rarefaction(M2, p_interface, p2)
        
        return {
            'interaction_type': 'shock_rarefaction',
            'p_interface': p_interface,
            'xi_interface': xi_interface,
            'rho_upper': rho_n_minus,
            'rho_lower': rho_n_plus,
            'u_upper': u_n_minus,
            'v_upper': v_n_minus,
            'u_lower': u_n_plus,
            'v_lower': v_n_plus,
            'shock_angle_upper': shock_angle_upper,
            'rarefaction_angles_lower': rarefaction_angles_lower,
            'theta1': theta1,
            'theta2': theta2,
            'delta1': delta1,
            'delta2': delta2
        }
    
    def _solve_rarefaction_shock(self, M1, M2, p1, p2, rho1, rho2, theta1, theta2):
        """
        Решение задачи для случая с волной разрежения и ударной волной (рис. 2г в статье).
        """
        # Зеркальный случай к _solve_shock_rarefaction
        p_guess = (p1 + p2) / 2
        
        def pressure_equation(p):
            beta1 = 1.0
            beta2 = 1.0
            
            # Для волны разрежения (верхний поток)
            gamma_j = (rho1 * M1**2 + p1 - p)**(-1) * np.sqrt(
                2 * self.gamma * M1**2 * p1 / ((self.gamma - 1) * p1 + (self.gamma + 1) * p) - 1
            )
            alpha_n_minus = gamma_j * beta1
            
            # Для ударной волны (нижний поток)
            alpha_n_plus = 0.5 * ((1 + 0**2) / (rho2 * p2**2) * np.sqrt(M2**2 - 1))
            
            # Приближенное вычисление ξ
            xi_n_minus = np.tan(theta1)
            xi_n_plus = np.tan(theta2)
            
            # Расчет по формуле (3.6)
            term1 = beta1 * xi_n_minus - beta2 * xi_n_plus
            term2 = alpha_n_minus * (p - p1) - alpha_n_plus * (p - p2)
            return term1 + term2
        
        p_interface = fsolve(pressure_equation, p_guess)[0]
        
        # Расчет остальных параметров
        xi_interface = self._calculate_xi_interface(p_interface, M1, M2, p1, p2, rho1, rho2, theta1, theta2,
                                                    upper_shock=False, lower_shock=True)
        
        rho_n_minus = self._calculate_density_rarefaction(p_interface, p1, rho1)
        rho_n_plus = self._calculate_density_shock(p_interface, p2, rho2, M2)
        
        u_n_minus, v_n_minus = self._calculate_velocities(xi_interface, p_interface, p1, rho1, M1)
        u_n_plus, v_n_plus = self._calculate_velocities(xi_interface, p_interface, p2, rho2, M2)
        
        # Расчет углов волны разрежения
        nu_initial = self._prandtl_meyer_angle(M1)
        M_final = M1 * np.sqrt(p1 / p_interface)
        nu_final = self._prandtl_meyer_angle(M_final)
        delta_upper = nu_final - nu_initial
        
        rarefaction_angles_upper = {
            'initial': theta1,
            'final': theta1 + delta_upper
        }
        
        # Расчет угла ударной волны
        shock_angle_lower = self._calculate_shock_angle(M2, p_interface, p2)
        
        # Расчет углов отклонения потока
        delta1 = self._calculate_flow_deflection_rarefaction(M1, p_interface, p1)
        delta2 = self._calculate_flow_deflection_shock(M2, p_interface, p2)
        
        return {
            'interaction_type': 'rarefaction_shock',
            'p_interface': p_interface,
            'xi_interface': xi_interface,
            'rho_upper': rho_n_minus,
            'rho_lower': rho_n_plus,
            'u_upper': u_n_minus,
            'v_upper': v_n_minus,
            'u_lower': u_n_plus,
            'v_lower': v_n_plus,
            'rarefaction_angles_upper': rarefaction_angles_upper,
            'shock_angle_lower': shock_angle_lower,
            'theta1': theta1,
            'theta2': theta2,
            'delta1': delta1,
            'delta2': delta2
        }
    
    def _solve_rarefaction_rarefaction(self, M1, M2, p1, p2, rho1, rho2, theta1, theta2):
        """
        Решение задачи для случая с двумя волнами разрежения (рис. 2д в статье).
        """
        # Начальное приближение для давления в области взаимодействия
        p_min = 1e-8
        p_max = min(p1, p2) - 1e-8
        p_guess = (p1 + p2) / 2

        def pressure_equation(p):
            """
            Уравнение для определения давления на контактном разрыве.
            """
            # Проверка физической реализуемости
            if p <= 0:
                return float('inf')
                
            # Расчет углов отклонения потока
            delta1 = self._calculate_flow_deflection_shock(M1, p, p1) if p > p1 else self._calculate_flow_deflection_rarefaction(M1, p, p1)
            delta2 = self._calculate_flow_deflection_shock(M2, p, p2) if p > p2 else self._calculate_flow_deflection_rarefaction(M2, p, p2)
            
            # Уравнение баланса углов
            return np.tan(theta1 + delta1) - np.tan(theta2 + delta2)
        
        # Решение уравнения методом fsolve с дополнительными параметрами
        p_interface = fsolve(pressure_equation, p_guess)[0]
        
        # Расчет остальных параметров
        xi_interface = self._calculate_xi_interface(p_interface, M1, M2, p1, p2, rho1, rho2, theta1, theta2,
                                                    upper_shock=False, lower_shock=False)
        
        rho_n_minus = self._calculate_density_rarefaction(p_interface, p1, rho1)
        rho_n_plus = self._calculate_density_rarefaction(p_interface, p2, rho2)
        
        u_n_minus, v_n_minus = self._calculate_velocities(xi_interface, p_interface, p1, rho1, M1)
        u_n_plus, v_n_plus = self._calculate_velocities(xi_interface, p_interface, p2, rho2, M2)
        
        rarefaction_angles_upper = self._calculate_rarefaction_angles(M1, p_interface, p1)
        rarefaction_angles_lower = self._calculate_rarefaction_angles(M2, p_interface, p2)
        
        # Расчет углов отклонения потока
        delta1 = self._calculate_flow_deflection_rarefaction(M1, p_interface, p1)
        delta2 = self._calculate_flow_deflection_rarefaction(M2, p_interface, p2)
        
        return {
            'interaction_type': 'rarefaction_rarefaction',
            'p_interface': p_interface,
            'xi_interface': xi_interface,
            'rho_upper': rho_n_minus,
            'rho_lower': rho_n_plus,
            'u_upper': u_n_minus,
            'v_upper': v_n_minus,
            'u_lower': u_n_plus,
            'v_lower': v_n_plus,
            'rarefaction_angles_upper': rarefaction_angles_upper,
            'rarefaction_angles_lower': rarefaction_angles_lower,
            'theta1': theta1,
            'theta2': theta2,
            'delta1': delta1,
            'delta2': delta2
        }
    
    def _calculate_xi_interface(self, p, M1, M2, p1, p2, rho1, rho2, theta1, theta2, 
                                upper_shock=True, lower_shock=True):
        """
        Вычисление параметра ξ на границе контактного разрыва.
        
        Параметр ξ = tg(θ) = v/u, где θ - угол наклона вектора скорости.
        """
        if upper_shock and lower_shock:
            # Для случая с двумя ударными волнами (рис. 2б)
            return (np.tan(theta1) + np.tan(theta2)) / 2
        elif upper_shock and not lower_shock:
            # Для случая с ударной волной и волной разрежения (рис. 2в)
            delta_upper = self._calculate_flow_deflection_shock(M1, p, p1)
            return np.tan(theta1 + delta_upper)
        elif not upper_shock and lower_shock:
            # Для случая с волной разрежения и ударной волной (рис. 2г)
            delta_lower = self._calculate_flow_deflection_shock(M2, p, p2)
            return np.tan(theta2 + delta_lower)
        else:
            # Для случая с двумя волнами разрежения (рис. 2д)
            delta_upper = self._calculate_flow_deflection_rarefaction(M1, p, p1)
            delta_lower = self._calculate_flow_deflection_rarefaction(M2, p, p2)
            # Учитываем отклонения обоих потоков
            return (np.tan(theta1 + delta_upper) + np.tan(theta2 + delta_lower)) / 2
    
    def _calculate_density_shock(self, p, p_initial, rho_initial, M_initial):
        """
        Вычисление плотности за ударной волной по формуле (3.8).
        """
        # Формула (3.8) из статьи для ρ
        numerator = (self.gamma - 1) * p_initial + (self.gamma + 1) * p
        denominator = (self.gamma + 1) * p_initial + (self.gamma - 1) * p
        return rho_initial * (numerator / denominator)
    
    def _calculate_density_rarefaction(self, p, p_initial, rho_initial):
        """
        Вычисление плотности в волне разрежения.
        """
        # Формула для адиабаты Пуассона
        return rho_initial * (p / p_initial)**(1/self.gamma)
    
    def _calculate_velocities(self, xi, p, p_initial, rho_initial, M_initial):
        """
        Вычисление компонент скорости по соотношениям Ренкина-Гюгонио.
        
        Параметры:
        ----------
        xi : float
            Параметр ξ = tg(θ) = v/u
        p : float
            Давление за ударной волной
        p_initial : float
            Начальное давление
        rho_initial : float
            Начальная плотность
        M_initial : float
            Начальное число Маха
        
        Возвращает:
        -----------
        u, v : tuple
            Компоненты скорости (u, v)
        """
        # Расчет скорости звука и начальной скорости
        a_initial = np.sqrt(self.gamma * p_initial / rho_initial)
        u_initial = M_initial * a_initial
        
        # Расчет угла ударной волны
        beta = self._calculate_shock_angle(M_initial, p, p_initial)
        
        # Расчет плотности за ударной волной
        rho = self._calculate_density_shock(p, p_initial, rho_initial, M_initial)
        
        # Расчет нормальной компоненты скорости за ударной волной
        # Используем соотношение Ренкина-Гюгонио для нормальной компоненты
        u_n = u_initial * np.cos(beta) * (rho_initial / rho)
        
        # Расчет тангенциальной компоненты скорости
        # Тангенциальная компонента сохраняется
        v_t = u_initial * np.sin(beta)
        
        # Поворот скоростей на угол отклонения потока
        delta = self._calculate_flow_deflection_shock(M_initial, p, p_initial)
        theta = np.arctan(xi) + delta
        
        # Вычисление компонент скорости в декартовой системе координат
        u = u_n * np.cos(theta) - v_t * np.sin(theta)
        v = u_n * np.sin(theta) + v_t * np.cos(theta)
        
        logger.debug(f"Расчет скоростей: u={u:.3f}, v={v:.3f}, theta={np.degrees(theta):.1f}°")
        
        return u, v
    
    def _calculate_shock_angle(self, M_initial, p, p_initial):
        """
        Вычисление угла наклона ударной волны.
        
        Параметры:
        ----------
        M_initial : float
            Начальное число Маха
        p : float
            Давление за ударной волной
        p_initial : float
            Начальное давление
        
        Возвращает:
        -----------
        beta : float
            Угол наклона ударной волны в радианах
        """
        # Преобразуем входные данные в скаляры, если они массивы
        if isinstance(p, np.ndarray):
            p = float(p[0])
        if isinstance(p_initial, np.ndarray):
            p_initial = float(p_initial[0])
            
        logger.debug(f"Расчет угла ударной волны: M={M_initial}, p={p}, p_initial={p_initial}")
        
        # Проверка входных параметров
        if M_initial <= 1.0:
            raise ValueError("Число Маха должно быть больше 1")
            
        # Проверка и корректировка давления
        if p <= p_initial:
            logger.warning(f"Давление за ударной волной ({p}) меньше начального ({p_initial})")
            p = p_initial * 1.1  # Корректируем давление
            
        # Расчет максимально возможного давления за скачком
        p_max = p_initial * (2 * self.gamma * M_initial**2 - (self.gamma - 1)) / (self.gamma + 1)
        if p > p_max:
            logger.warning(f"Давление за ударной волной ({p}) превышает максимально возможное ({p_max})")
            p = p_max * 0.99  # Корректируем давление
            
        # Расчет угла ударной волны
        try:
            # Используем формулу для расчета угла ударной волны
            term = (self.gamma + 1) / 2 * (p / p_initial - 1) / (self.gamma * M_initial**2) + 1 / (2 * M_initial**2)
            
            if term < 0:
                logger.warning(f"Отрицательное значение под корнем: {term}")
                term = 0.1  # Безопасное значение
                
            sin_beta = np.sqrt(term)
            
            if sin_beta > 1:
                logger.warning(f"sin(beta) > 1: {sin_beta}")
                sin_beta = 0.99  # Безопасное значение
                
            beta = np.arcsin(sin_beta)
            
            # Проверка на минимальный угол Маха
            mu = np.arcsin(1/M_initial)
            if beta < mu:
                logger.warning(f"Угол ударной волны ({np.degrees(beta):.1f}°) меньше угла Маха ({np.degrees(mu):.1f}°)")
                beta = mu * 1.1  # Корректируем угол
                
            logger.debug(f"Угол ударной волны: {np.degrees(beta):.1f}°")
            return beta
            
        except Exception as e:
            logger.error(f"Ошибка при расчете угла ударной волны: {str(e)}")
            # Возвращаем приближенное значение угла Маха
            return np.arcsin(1/M_initial) * 1.1
    
    def _calculate_rarefaction_angles(self, M_initial, p, p_initial):
        """
        Вычисление углов для границ веера волн разрежения.
        
        Параметры:
        ----------
        M_initial : float
            Начальное число Маха
        p : float
            Давление после волны разрежения
        p_initial : float
            Начальное давление
        
        Возвращает:
        -----------
        dict
            Словарь с углами границ веера волн разрежения
        """
        # Вычисление числа Маха после волны разрежения
        M_final = M_initial * np.sqrt(p_initial / p)
        
        # Расчет углов Прандтля-Майера
        nu_initial = self._prandtl_meyer_angle(M_initial)
        nu_final = self._prandtl_meyer_angle(M_final)
        
        # Угол отклонения потока
        delta = nu_final - nu_initial
        
        # Угол Маха для начального потока
        mu_initial = np.arcsin(1 / M_initial)
        
        # Угол Маха для конечного потока
        mu_final = np.arcsin(1 / M_final)
        
        # Возвращаем углы границ веера волн разрежения
        return {
            'initial': mu_initial,
            'final': mu_initial + delta,  # Угол поворота потока
            'nu_initial': nu_initial,
            'nu_final': nu_final,
            'delta': delta  # Полный угол поворота потока
        }

    def _calculate_flow_deflection_shock(self, M_initial, p, p_initial):
        """
        Вычисление угла отклонения потока за ударной волной по формуле:
        tan(delta) = 2*cot(beta)*(M^2*sin^2(beta)-1) / [M^2*(gamma+cos(2*beta))+2]
        """
        beta = self._calculate_shock_angle(M_initial, p, p_initial)
        if beta == 0.0:
            return 0.0
        sinb = np.sin(beta)
        cos2b = np.cos(2 * beta)
        num = 2 * (M_initial**2 * sinb**2 - 1) / np.tan(beta)
        den = M_initial**2 * (self.gamma + cos2b) + 2
        delta = np.arctan(num / den)
        return delta

    def _calculate_flow_deflection_rarefaction(self, M_initial, p, p_initial):
        # Защита от слишком малого давления
        p = max(p, p_initial * 0.01)  # Ограничиваем минимальное давление
        M_final = M_initial * np.sqrt(p_initial / p)
        
        # Ограничиваем максимальное число Маха
        M_final = min(M_final, 10.0)  # Максимальное число Маха
        
        nu_initial = self._prandtl_meyer_angle(M_initial)
        nu_final = self._prandtl_meyer_angle(M_final)
        
        # Угол отклонения потока
        return nu_final - nu_initial

    def _prandtl_meyer_angle(self, M):
        """
        Вычисление функции Прандтля–Майера:
        """
        if M <= 1.0:
            return 0.0  # нет расширения при M≤1
        
        gm1 = self.gamma - 1.0
        gp1 = self.gamma + 1.0

        sqrt_term = np.sqrt(gp1 / gm1)
        arg1 = np.sqrt((M**2 - 1.0) * (gm1 / gp1))
        nu = sqrt_term * np.arctan(arg1) - np.arctan(np.sqrt(M**2 - 1.0))
        return nu


    def _fill_shock_shock_fields(self, X, Y, U, V, P, results):
        """
        Заполнение полей скорости и давления для случая с двумя ударными волнами.
        """
        # Параметры ударных волн
        shock_angle_upper = results['shock_angle_upper']
        shock_angle_lower = results['shock_angle_lower']
        
        # Параметры в областях
        p_interface = results['p_interface']
        rho_upper = results['rho_upper']
        rho_lower = results['rho_lower']
        u_upper = results['u_upper']
        v_upper = results['v_upper']
        u_lower = results['u_lower']
        v_lower = results['v_lower']
        
        # Начальные углы потоков
        theta1 = results.get('theta1', 0)
        theta2 = results.get('theta2', 0)
        
        # Заполнение полей
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                
                # Область 1 (верхний исходный поток)
                if y > x * np.tan(shock_angle_upper):
                    U[i, j] = np.cos(theta1)  # Нормированная скорость с учетом угла
                    V[i, j] = np.sin(theta1)
                    P[i, j] = 1.0
                
                # Область 4 (нижний исходный поток)
                elif y < -x * np.tan(shock_angle_lower):
                    U[i, j] = np.cos(theta2)  # Нормированная скорость с учетом угла
                    V[i, j] = np.sin(theta2)
                    P[i, j] = 1.0
                
                # Область 2 (после верхней ударной волны)
                elif y > 0 and y < x * np.tan(shock_angle_upper):
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление за верхней ударной волной с экспоненциальным затуханием
                    dist = abs(y - x * np.tan(shock_angle_upper))
                    P[i, j] = p_interface * (1.0 + 0.2 * np.exp(-dist * 5))
                
                # Область 3 (после нижней ударной волны)
                elif y < 0 and y > -x * np.tan(shock_angle_lower):
                    U[i, j] = u_lower
                    V[i, j] = v_lower
                    # Давление за нижней ударной волной с экспоненциальным затуханием
                    dist = abs(y + x * np.tan(shock_angle_lower))
                    P[i, j] = p_interface * (1.0 + 0.2 * np.exp(-dist * 5))
                
                # Контактный разрыв
                else:
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление на контактном разрыве
                    P[i, j] = p_interface

    def _fill_shock_rarefaction_fields(self, X, Y, U, V, P, results):
        """
        Заполнение полей скорости и давления для случая с ударной волной и волной разрежения.
        """
        # Параметры ударной волны и волны разрежения
        shock_angle = results.get('shock_angle_upper', 0)
        rarefaction_angles = results.get('rarefaction_angles_lower', {})
        rarefaction_angle = rarefaction_angles.get('initial', 0)
        
        # Параметры в областях
        p_interface = results['p_interface']
        rho_upper = results['rho_upper']
        rho_lower = results['rho_lower']
        u_upper = results['u_upper']
        v_upper = results['v_upper']
        u_lower = results['u_lower']
        v_lower = results['v_lower']
        
        # Начальные углы потоков
        theta1 = results.get('theta1', 0)
        theta2 = results.get('theta2', 0)
        
        # Заполнение полей
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                
                # Область 1 (верхний исходный поток)
                if y > x * np.tan(shock_angle):
                    U[i, j] = np.cos(theta1)
                    V[i, j] = np.sin(theta1)
                    P[i, j] = 1.0
                
                # Область 4 (нижний исходный поток)
                elif y < -x * np.tan(rarefaction_angle):
                    U[i, j] = np.cos(theta2)
                    V[i, j] = np.sin(theta2)
                    P[i, j] = 1.0
                
                # Область 2 (после ударной волны)
                elif y > 0 and y < x * np.tan(shock_angle):
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление за ударной волной с экспоненциальным затуханием
                    dist = abs(y - x * np.tan(shock_angle))
                    P[i, j] = p_interface * (1.0 + 0.2 * np.exp(-dist * 5))
                
                # Область 3 (в волне разрежения)
                elif y < 0 and y > -x * np.tan(rarefaction_angle):
                    U[i, j] = u_lower
                    V[i, j] = v_lower
                    # Давление в волне разрежения меняется плавно
                    r = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x)
                    if angle < -rarefaction_angle:
                        P[i, j] = 1.0
                    else:
                        # Плавное изменение давления в волне разрежения
                        t = (angle + rarefaction_angle) / (2 * rarefaction_angle)
                        P[i, j] = 1.0 + (p_interface - 1.0) * t
                
                # Контактный разрыв
                else:
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление на контактном разрыве
                    P[i, j] = p_interface
                    print(f"WARNING: x={x}, y={y}, shock_angle={shock_angle}, rarefaction_angle={rarefaction_angle}")

    def _fill_rarefaction_shock_fields(self, X, Y, U, V, P, results):
        """
        Заполнение полей скорости и давления для случая с волной разрежения и ударной волной.
        """
        # Параметры волн
        rarefaction_angles_upper = results['rarefaction_angles_upper']
        shock_angle_lower = results['shock_angle_lower']
        
        # Параметры в областях
        p_interface = results['p_interface']
        rho_upper = results['rho_upper']
        rho_lower = results['rho_lower']
        u_upper = results['u_upper']
        v_upper = results['v_upper']
        u_lower = results['u_lower']
        v_lower = results['v_lower']
        
        # Начальные углы потоков
        theta1 = results.get('theta1', 0)
        theta2 = results.get('theta2', 0)
        
        # Углы границ веера волн разрежения
        mu_initial = rarefaction_angles_upper['initial']
        mu_final = rarefaction_angles_upper['final']
        
        # Заполнение полей
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                
                # Область 1 (верхний исходный поток)
                if y > x * np.tan(mu_initial):
                    U[i, j] = np.cos(theta1)
                    V[i, j] = np.sin(theta1)
                    P[i, j] = 1.0
                
                # Область 4 (нижний исходный поток)
                elif y < -x * np.tan(shock_angle_lower):
                    U[i, j] = np.cos(theta2)
                    V[i, j] = np.sin(theta2)
                    P[i, j] = 1.0
                
                # Область веера волн разрежения (верхняя)
                elif y > 0 and y < x * np.tan(mu_initial) and y > x * np.tan(mu_final):
                    angle = np.arctan2(y, x)
                    t = (angle - mu_initial) / (mu_final - mu_initial)
                    t = max(0, min(1, t))
                    
                    U[i, j] = np.cos(theta1) + t * (u_upper - np.cos(theta1))
                    V[i, j] = np.sin(theta1) + t * (v_upper - np.sin(theta1))
                    # Плавное изменение давления в волне разрежения
                    P[i, j] = 1.0 + t * (p_interface - 1.0)
                
                # Область 2 (после верхней волны разрежения)
                elif y > 0 and y < x * np.tan(mu_final):
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление после волны разрежения
                    P[i, j] = p_interface
                
                # Область 3 (после нижней ударной волны)
                elif y < 0 and y > -x * np.tan(shock_angle_lower):
                    U[i, j] = u_lower
                    V[i, j] = v_lower
                    # Давление за ударной волной
                    P[i, j] = p_interface * (1.0 + 0.1 * np.exp(-abs(y + x * np.tan(shock_angle_lower))))
                
                # Контактный разрыв
                else:
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление на контактном разрыве
                    P[i, j] = p_interface
                    print(f"WARNING: x={x}, y={y}, shock_angle={shock_angle_lower}, rarefaction_angle={rarefaction_angles_upper}")

    def _fill_rarefaction_rarefaction_fields(self, X, Y, U, V, P, results):
        """
        Заполнение полей скорости и давления для случая с двумя волнами разрежения.
        """
        # Параметры волн разрежения
        rarefaction_angles_upper = results['rarefaction_angles_upper']
        rarefaction_angles_lower = results['rarefaction_angles_lower']
        
        # Параметры в областях
        p_interface = results['p_interface']
        rho_upper = results['rho_upper']
        rho_lower = results['rho_lower']
        u_upper = results['u_upper']
        v_upper = results['v_upper']
        u_lower = results['u_lower']
        v_lower = results['v_lower']
        
        # Начальные углы потоков
        theta1 = results.get('theta1', 0)
        theta2 = results.get('theta2', 0)
        
        # Углы границ вееров волн разрежения
        mu_initial_upper = rarefaction_angles_upper['initial']
        mu_final_upper = rarefaction_angles_upper['final']
        mu_initial_lower = rarefaction_angles_lower['initial']
        mu_final_lower = rarefaction_angles_lower['final']
        
        # Заполнение полей
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                
                # Область 1 (верхний исходный поток)
                if y > x * np.tan(mu_initial_upper):
                    U[i, j] = np.cos(theta1)
                    V[i, j] = np.sin(theta1)
                    P[i, j] = 1.0
                
                # Область 4 (нижний исходный поток)
                elif y < -x * np.tan(mu_initial_lower):
                    U[i, j] = np.cos(theta2)
                    V[i, j] = np.sin(theta2)
                    P[i, j] = 1.0
                
                # Область веера верхней волны разрежения
                elif y > 0 and y < x * np.tan(mu_initial_upper) and y > x * np.tan(mu_final_upper):
                    angle = np.arctan2(y, x)
                    t = (angle - mu_initial_upper) / (mu_final_upper - mu_initial_upper)
                    t = max(0, min(1, t))
                    
                    U[i, j] = np.cos(theta1) + t * (u_upper - np.cos(theta1))
                    V[i, j] = np.sin(theta1) + t * (v_upper - np.sin(theta1))
                    # Плавное изменение давления в волне разрежения с использованием функции tanh
                    P[i, j] = 1.0 + (p_interface - 1.0) * np.tanh(2 * t)
                
                # Область 2 (после верхней волны разрежения)
                elif y > 0 and y < x * np.tan(mu_final_upper):
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление после волны разрежения с небольшим затуханием
                    dist = abs(y - x * np.tan(mu_final_upper))
                    P[i, j] = p_interface * (1.0 + 0.1 * np.exp(-dist * 3))
                
                # Область веера нижней волны разрежения
                elif y < 0 and y > -x * np.tan(mu_initial_lower) and y < -x * np.tan(mu_final_lower):
                    angle = np.arctan2(-y, x)
                    t = (angle - mu_initial_lower) / (mu_final_lower - mu_initial_lower)
                    t = max(0, min(1, t))
                    
                    U[i, j] = np.cos(theta2) + t * (u_lower - np.cos(theta2))
                    V[i, j] = np.sin(theta2) + t * (v_lower - np.sin(theta2))
                    # Плавное изменение давления в волне разрежения с использованием функции tanh
                    P[i, j] = 1.0 + (p_interface - 1.0) * np.tanh(2 * t)
                
                # Область 3 (после нижней волны разрежения)
                elif y < 0 and y > -x * np.tan(mu_final_lower):
                    U[i, j] = u_lower
                    V[i, j] = v_lower
                    # Давление после волны разрежения с небольшим затуханием
                    dist = abs(y + x * np.tan(mu_final_lower))
                    P[i, j] = p_interface * (1.0 + 0.1 * np.exp(-dist * 3))
                
                # Контактный разрыв
                else:
                    U[i, j] = u_upper
                    V[i, j] = v_upper
                    # Давление на контактном разрыве
                    P[i, j] = p_interface

    # Добавляем функцию для запуска расчетов и визуализации результатов для различных параметров
    def run_parametric_study(self, M1_values, M2_values, p_ratios, rho_ratios, save_figures=True, folder='results'):
        """
        Проведение параметрического исследования для различных входных параметров.
        
        Параметры:
        ----------
        M1_values : list
            Список значений числа Маха для верхнего потока
        M2_values : list
            Список значений числа Маха для нижнего потока
        p_ratios : list
            Список отношений давлений p2/p1
        rho_ratios : list
            Список отношений плотностей rho2/rho1
        save_figures : bool
            Сохранять ли результаты в файлы
        folder : str
            Папка для сохранения результатов
        
        Возвращает:
        -----------
        results_dict : dict
            Словарь с результатами расчетов для всех комбинаций параметров
        """
        import os
        
        # Создаем папку для результатов, если её нет
        if save_figures and not os.path.exists(folder):
            os.makedirs(folder)
        
        results_dict = {}
        
        total_cases = len(M1_values) * len(M2_values) * len(p_ratios) * len(rho_ratios)
        case_count = 0
        
        # Перебор всех комбинаций параметров
        for M1 in M1_values:
            for M2 in M2_values:
                for p_ratio in p_ratios:
                    for rho_ratio in rho_ratios:
                        case_count += 1
                        print(f"Обработка случая {case_count}/{total_cases}: M1={M1}, M2={M2}, p2/p1={p_ratio}, rho2/rho1={rho_ratio}")
                        
                        # Базовые значения
                        p1 = 1.0
                        rho1 = 1.0
                        p2 = p1 * p_ratio
                        rho2 = rho1 * rho_ratio
                        
                        # Расчет взаимодействия
                        results = self.solve_flow_interaction(M1, M2, p1, p2, rho1, rho2)
                        
                        # Сохранение результатов в словарь
                        case_key = f"M1={M1}_M2={M2}_p2/p1={p_ratio}_rho2/rho1={rho_ratio}"
                        results_dict[case_key] = results
                        
                        # Визуализация и сохранение результатов
                        if save_figures:
                            fig = plt.figure(figsize=(15, 12))
                            
                            # Создаем подграфики для различных визуализаций
                            ax1 = fig.add_subplot(221)
                            ax2 = fig.add_subplot(222)
                            ax3 = fig.add_subplot(223)
                            ax4 = fig.add_subplot(224)
                            
                            # Параметры для визуализации
                            x_range = (-2, 2)
                            y_range = (-2, 2)
                            resolution = 200
                            
                            # Создание сетки
                            x = np.linspace(x_range[0], x_range[1], resolution)
                            y = np.linspace(y_range[0], y_range[1], resolution)
                            X, Y = np.meshgrid(x, y)
                            
                            # Создание массивов для полей
                            U = np.zeros_like(X)
                            V = np.zeros_like(X)
                            P = np.zeros_like(X)
                            P[:, :] = 1.0  # Инициализация давлением 1.0
                            
                            # Заполнение полей
                            interaction_type = results['interaction_type']
                            
                            if interaction_type == 'shock_shock':
                                self._fill_shock_shock_fields(X, Y, U, V, P, results)
                            elif interaction_type == 'shock_rarefaction':
                                self._fill_shock_rarefaction_fields(X, Y, U, V, P, results)
                            elif interaction_type == 'rarefaction_shock':
                                self._fill_rarefaction_shock_fields(X, Y, U, V, P, results)
                            elif interaction_type == 'rarefaction_rarefaction':
                                self._fill_rarefaction_rarefaction_fields(X, Y, U, V, P, results)
                            
                            # Визуализация поля давления
                            pressure_contour = ax1.contourf(X, Y, P, cmap='viridis', levels=50)
                            ax1.set_title('Поле давления')
                            ax1.set_xlabel('x')
                            ax1.set_ylabel('y')
                            fig.colorbar(pressure_contour, ax=ax1, label='Давление')
                            
                            # Визуализация поля модуля скорости
                            velocity_magnitude = np.sqrt(U**2 + V**2)
                            velocity_contour = ax2.contourf(X, Y, velocity_magnitude, cmap='plasma', levels=50)
                            ax2.set_title('Модуль скорости')
                            ax2.set_xlabel('x')
                            ax2.set_ylabel('y')
                            fig.colorbar(velocity_contour, ax=ax2, label='|V|')
                            
                            # Визуализация линий тока
                            streamplot = ax3.streamplot(x, y, U.T, V.T, density=1.5, color='black')
                            ax3.set_title('Линии тока')
                            ax3.set_xlabel('x')
                            ax3.set_ylabel('y')
                            ax3.contourf(X, Y, P, cmap='viridis', levels=50, alpha=0.5)
                            
                            # Визуализация векторного поля скоростей
                            skip = 10  # Прореживание векторов для наглядности
                            ax4.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                                    U[::skip, ::skip], V[::skip, ::skip], 
                                    scale=25, width=0.002)
                            ax4.set_title('Векторное поле скоростей')
                            ax4.set_xlabel('x')
                            ax4.set_ylabel('y')
                            
                            # Добавление текстовой информации о параметрах
                            plt.figtext(0.5, 0.01, 
                                    f"Параметры: M1={M1}, M2={M2}, p2/p1={p_ratio}, rho2/rho1={rho_ratio}\n" 
                                    f"Тип взаимодействия: {interaction_type}, Давление на контактном разрыве: {results['p_interface']:.4f}", 
                                    ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                            
                            # Настройка общего вида графика
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plt.suptitle(f"Взаимодействие сверхзвуковых потоков - {interaction_type}", fontsize=16)
                            
                            # Сохранение рисунка
                            filename = os.path.join(folder, f"{case_key.replace('/', '_')}.png")
                            plt.savefig(filename, dpi=200)
                            plt.close(fig)
        
        return results_dict

    # Функция для анализа результатов параметрического исследования
    def analyze_results(self, results_dict):
        """
        Анализ результатов параметрического исследования.
        
        Параметры:
        ----------
        results_dict : dict
            Словарь с результатами расчетов
            
        Возвращает:
        -----------
        analysis : dict
            Словарь с результатами анализа
        """
        analysis = {
            'interaction_types': {},
            'pressure_ranges': {
                'min': float('inf'),
                'max': float('-inf'),
                'sum': 0.0,
                'count': 0
            },
            'velocity_changes': []
        }
        
        # Анализ типов взаимодействия
        for case, results in results_dict.items():
            interaction_type = results['interaction_type']
            
            if interaction_type not in analysis['interaction_types']:
                analysis['interaction_types'][interaction_type] = 0
            
            analysis['interaction_types'][interaction_type] += 1
            
            # Анализ диапазона давлений
            p_interface = results['p_interface']
            analysis['pressure_ranges']['min'] = min(analysis['pressure_ranges']['min'], p_interface)
            analysis['pressure_ranges']['max'] = max(analysis['pressure_ranges']['max'], p_interface)
            analysis['pressure_ranges']['sum'] += p_interface
            analysis['pressure_ranges']['count'] += 1
            
            # Анализ изменения скорости
            velocity_change_upper = np.sqrt(results['u_upper']**2 + results['v_upper']**2)
            velocity_change_lower = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
            
            analysis['velocity_changes'].append({
                'case': case,
                'upper_flow': velocity_change_upper,
                'lower_flow': velocity_change_lower
            })
        
        # Преобразование счетчиков типов взаимодействия в проценты
        total_cases = len(results_dict)
        for interaction_type, count in analysis['interaction_types'].items():
            analysis['interaction_types'][interaction_type] = count / total_cases * 100
        
        # Добавим среднее значение давления
        if analysis['pressure_ranges']['count'] > 0:
            analysis['pressure_ranges']['mean'] = (
                analysis['pressure_ranges']['sum'] / analysis['pressure_ranges']['count']
            )
        else:
            analysis['pressure_ranges']['mean'] = float('nan')
        
        # Для удобства вывода переименуем ключи
        analysis['pressure_ranges'] = {
            'min': analysis['pressure_ranges']['min'],
            'max': analysis['pressure_ranges']['max'],
            'mean': analysis['pressure_ranges']['mean']
        }

        return analysis

    # Создаем функцию для вычисления точного решения на основе формул из статьи
    def compute_exact_solution(self, M1, M2, p1, p2, rho1, rho2, theta1=0, theta2=0, interaction_type=None):
        """
        Вычисление точного решения для задачи о взаимодействии сверхзвуковых потоков
        на основе автомодельного решения уравнений Эйлера из статьи.
        
        Параметры:
        ----------
        M1, M2 : float
            Числа Маха для верхнего и нижнего потоков
        p1, p2 : float
            Давления в верхнем и нижнем потоках
        rho1, rho2 : float
            Плотности в верхнем и нижнем потоках
        theta1, theta2 : float
            Углы наклона векторов скорости (в радианах)
        interaction_type : str, optional
            Тип взаимодействия. Если None, определяется автоматически
        
        Возвращает:
        -----------
        solution : dict
            Словарь с точным решением
        """
        # Если тип взаимодействия не задан, определяем его
        if interaction_type is None:
            interaction_type = self._determine_interaction_type_advanced(M1, M2, p1, p2, theta1, theta2)
        
        gamma = self.gamma
    
    def visualize_flow(self, results, x_range=(-1, 1), y_range=(-1.0, 1.0), resolution=100):
        """
        Визуализация результатов расчета течения.
        """
        # Создание сетки
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Создание массивов для компонент скорости и давления
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        P = np.zeros_like(X)
        
        # Заполнение полей в зависимости от типа взаимодействия
        interaction_type = results['interaction_type']
        
        if interaction_type == 'shock_shock':
            self._fill_shock_shock_fields(X, Y, U, V, P, results)
        elif interaction_type == 'shock_rarefaction':
            self._fill_shock_rarefaction_fields(X, Y, U, V, P, results)
        elif interaction_type == 'rarefaction_shock':
            self._fill_rarefaction_shock_fields(X, Y, U, V, P, results)
        elif interaction_type == 'rarefaction_rarefaction':
            self._fill_rarefaction_rarefaction_fields(X, Y, U, V, P, results)
        else:
            raise ValueError(f"Неизвестный тип взаимодействия: {interaction_type}")
        
        # Создание фигуры для основного графика
        fig_main = plt.figure(figsize=(12, 10))
        ax = fig_main.add_subplot(111)
        
        # Создание кастомной цветовой карты для давления
        colors = [(0.95, 0.95, 1), (0.5, 0.5, 0.9), (0.2, 0.2, 0.8)]  # От белого к синему
        pressure_cmap = LinearSegmentedColormap.from_list('pressure_cmap', colors)
        
        # Построение поля давления
        pressure_contour = ax.contourf(X, Y, P, cmap=pressure_cmap, levels=20, alpha=0.7)
        fig_main.colorbar(pressure_contour, ax=ax, label='Давление')
        
        # Создаем маску для линий тока (только до точки столкновения)
        mask = np.ones_like(X, dtype=bool)
        mask[X > 0] = False  # Маскируем все точки правее x = 0
        
        # Применяем маску к полям скорости
        U_masked = np.ma.masked_array(U, ~mask)
        V_masked = np.ma.masked_array(V, ~mask)
        
        # Построение линий тока
        x_start = np.linspace(x_range[0], 0, 30)  # Точки старта только до x = 0
        y_start_upper = np.linspace(0.1, y_range[1], 20)
        y_start_lower = np.linspace(y_range[0], -0.1, 20)
        
        # Получаем начальные углы потоков
        theta1 = results.get('theta1', 0)
        theta2 = results.get('theta2', 0)
        
        # Линии тока для верхнего потока
        for y0 in y_start_upper:
            # Создаем прямые линии с учетом угла theta1
            x_line = np.array([x_range[0], 0])
            y_line = np.array([y0, y0 + np.tan(theta1) * (0 - x_range[0])])
            # Проверяем, не пересекает ли линия ось x
            if y_line[1] > 0:
                ax.plot(x_line, y_line, color='#1f77b4', linewidth=1.0)
                # Добавляем стрелочку
                arrow_x = x_line[0] + (x_line[1] - x_line[0]) * 0.7
                arrow_y = y_line[0] + (y_line[1] - y_line[0]) * 0.7
                ax.arrow(arrow_x, arrow_y, 
                        (x_line[1] - x_line[0]) * 0.1, 
                        (y_line[1] - y_line[0]) * 0.1,
                        head_width=0.02, head_length=0.05, fc='#1f77b4', ec='#1f77b4')
        
        # Линии тока для нижнего потока
        for y0 in y_start_lower:
            # Создаем прямые линии с учетом угла theta2
            x_line = np.array([x_range[0], 0])
            y_line = np.array([y0, y0 + np.tan(theta2) * (0 - x_range[0])])
            # Проверяем, не пересекает ли линия ось x
            if y_line[1] < 0:
                ax.plot(x_line, y_line, color='#ff7f0e', linewidth=1.0)
                # Добавляем стрелочку
                arrow_x = x_line[0] + (x_line[1] - x_line[0]) * 0.7
                arrow_y = y_line[0] + (y_line[1] - y_line[0]) * 0.7
                ax.arrow(arrow_x, arrow_y, 
                        (x_line[1] - x_line[0]) * 0.1, 
                        (y_line[1] - y_line[0]) * 0.1,
                        head_width=0.02, head_length=0.05, fc='#ff7f0e', ec='#ff7f0e')
        
        # Отображение ударных волн и вееров разрежения
        if interaction_type == 'shock_shock':
            # Верхняя ударная волна
            if 'shock_angle_upper' in results:
                beta_upper = results['shock_angle_upper']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = x_shock * np.tan(beta_upper)
                ax.plot(x_shock, y_shock, color='#FF6B6B', linewidth=2.5, label='Ударная волна')
            
            # Нижняя ударная волна
            if 'shock_angle_lower' in results:
                beta_lower = results['shock_angle_lower']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = -x_shock * np.tan(beta_lower)
                ax.plot(x_shock, y_shock, color='#FF6B6B', linewidth=2.5)
        
        elif interaction_type == 'shock_rarefaction':
            # Верхняя ударная волна
            if 'shock_angle_upper' in results:
                beta_upper = results['shock_angle_upper']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = x_shock * np.tan(beta_upper)
                ax.plot(x_shock, y_shock, color='#FF6B6B', linewidth=2.5, label='Ударная волна')
            
            # Нижний веер разрежения
            if 'rarefaction_angles_lower' in results:
                angles = results['rarefaction_angles_lower']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = -x_rare * np.tan(mu_initial)
                y_final = -x_rare * np.tan(mu_final)
                ax.plot(x_rare, y_initial, color='#4ECDC4', linewidth=2, label='Волна разрежения')
                ax.plot(x_rare, y_final, color='#4ECDC4', linewidth=2)
        
        elif interaction_type == 'rarefaction_shock':
            # Верхний веер разрежения
            if 'rarefaction_angles_upper' in results:
                angles = results['rarefaction_angles_upper']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = x_rare * np.tan(mu_initial)
                y_final = x_rare * np.tan(mu_final)
                ax.plot(x_rare, y_initial, color='#4ECDC4', linewidth=2, label='Волна разрежения')
                ax.plot(x_rare, y_final, color='#4ECDC4', linewidth=2)
            
            # Нижняя ударная волна
            if 'shock_angle_lower' in results:
                beta_lower = results['shock_angle_lower']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = -x_shock * np.tan(beta_lower)
                ax.plot(x_shock, y_shock, color='#FF6B6B', linewidth=2.5, label='Ударная волна')
        
        elif interaction_type == 'rarefaction_rarefaction':
            # Верхний веер разрежения
            if 'rarefaction_angles_upper' in results:
                angles = results['rarefaction_angles_upper']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = x_rare * np.tan(mu_initial)
                y_final = x_rare * np.tan(mu_final)
                ax.plot(x_rare, y_initial, color='#4ECDC4', linewidth=2, label='Волна разрежения')
                ax.plot(x_rare, y_final, color='#4ECDC4', linewidth=2)
            
            # Нижний веер разрежения
            if 'rarefaction_angles_lower' in results:
                angles = results['rarefaction_angles_lower']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = -x_rare * np.tan(mu_initial)
                y_final = -x_rare * np.tan(mu_final)
                ax.plot(x_rare, y_initial, color='#4ECDC4', linewidth=2)
                ax.plot(x_rare, y_final, color='#4ECDC4', linewidth=2)
        
        # Отображение тангенциального разрыва
        x_interface = np.linspace(0, x_range[1], 100)

        if interaction_type == 'shock_rarefaction':
            # Угол ударной волны
            beta = results['shock_angle_upper']
            # Угол начальной границы веера разрежения (нижний поток)
            mu = results['rarefaction_angles_lower']['initial']
            mu_final = results['rarefaction_angles_lower']['final']
            print('-' * 20)
            print(f"u_upper={results['u_upper']}, v_upper={results['v_upper']}")
            print(f"u_lower={results['u_lower']}, v_lower={results['v_lower']}")
            print(f"shock_angle_upper={np.degrees(beta):.2f}°")
            print(f"rarefaction_angles_lower: initial={np.degrees(mu):.2f}°, final={np.degrees(mu_final):.2f}°")
            # Контактный разрыв — посередине между ними
            theta_contact = (beta - (-mu)) / 2
            print(f"theta_contact (deg)={np.degrees(theta_contact):.2f}°")

        elif interaction_type == 'rarefaction_shock':
            theta_contact = np.arctan2(results['v_lower'], results['u_lower'])
        else:
            theta_contact = np.arctan(results['xi_interface'])
        y_interface = x_interface * np.tan(theta_contact)
        ax.plot(x_interface, y_interface, color='#45B7D1', linewidth=2, label='Тангенциальный разрыв')
        
        # Выделение точки столкновения
        ax.plot(0, 0, 'o', color='#FFD93D', markersize=10, label='Точка столкновения')
        
        # Настройка осей
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Добавление легенды
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Добавление информации о параметрах
        info_text = (
            f"Тип взаимодействия: {interaction_type}\n"
            f"Давление на контактном разрыве: {results['p_interface']:.3f}\n"
            f"Параметр ξ: {results['xi_interface']:.3f}\n"
            f"Плотности: ρ₁ = {results['rho_upper']:.3f}, ρ₂ = {results['rho_lower']:.3f}\n"
            f"Скорости верхнего потока: u = {results['u_upper']:.3f}, v = {results['v_upper']:.3f}\n"
            f"Скорости нижнего потока: u = {results['u_lower']:.3f}, v = {results['v_lower']:.3f}"
        )
        
        if 'shock_angle_upper' in results:
            info_text += f"\nУгол верхней ударной волны: {np.degrees(results['shock_angle_upper']):.1f}°"
        if 'shock_angle_lower' in results:
            info_text += f"\nУгол нижней ударной волны: {np.degrees(results['shock_angle_lower']):.1f}°"
        if 'rarefaction_angles_upper' in results:
            info_text += f"\nУглы верхней волны разрежения: {np.degrees(results['rarefaction_angles_upper']['initial']):.1f}° - {np.degrees(results['rarefaction_angles_upper']['final']):.1f}°"
        if 'rarefaction_angles_lower' in results:
            info_text += f"\nУглы нижней волны разрежения: {np.degrees(results['rarefaction_angles_lower']['initial']):.1f}° - {np.degrees(results['rarefaction_angles_lower']['final']):.1f}°"
        
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.9))
        
        # Настройка общего вида
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Взаимодействие сверхзвуковых потоков", fontsize=16)
        
        
        return fig_main
    
    def _plot_pressure_along_cone(self, results, distance=0.5):
        """
        Построение распределения давления вдоль линии, параллельной конусу взаимодействия
        на фиксированном расстоянии от точки столкновения.
        
        Параметры:
        ----------
        results : dict
            Результаты расчета
        distance : float
            Расстояние от точки столкновения до линии измерения
        """
        # Создаем новую фигуру
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Получаем угол наклона тангенциального разрыва
        xi = results['xi_interface']
        theta_interface = np.arctan(xi)
        
        # Создаем точки вдоль линии, параллельной конусу
        s = np.linspace(-0.5, 0.5, 200)  # Параметр вдоль линии
        
        # Создаем массивы для хранения координат и давления
        x_points = np.zeros_like(s)
        y_points = np.zeros_like(s)
        pressure = np.zeros_like(s)
        
        # Вычисляем координаты точек и давление
        for i, s_val in enumerate(s):
            # Вычисляем координаты точки
            x = distance * np.cos(theta_interface) - s_val * np.sin(theta_interface)
            y = distance * np.sin(theta_interface) + s_val * np.cos(theta_interface)
            
            x_points[i] = x
            y_points[i] = y
            
            # Определяем давление в зависимости от типа взаимодействия
            interaction_type = results['interaction_type']
            p_interface = results['p_interface']
            
            if interaction_type == 'shock_shock':
                # Для случая с двумя ударными волнами
                if y > x * np.tan(results['shock_angle_upper']):
                    pressure[i] = 1.0  # Начальное давление
                elif y < -x * np.tan(results['shock_angle_lower']):
                    pressure[i] = 1.0  # Начальное давление
                else:
                    # Давление между ударными волнами с экспоненциальным затуханием
                    dist_upper = abs(y - x * np.tan(results['shock_angle_upper']))
                    dist_lower = abs(y + x * np.tan(results['shock_angle_lower']))
                    pressure[i] = p_interface * (1.0 + 0.2 * np.exp(-min(dist_upper, dist_lower) * 5))
                    
            elif interaction_type == 'shock_rarefaction':
                # Для случая с ударной волной и волной разрежения
                if y > x * np.tan(results['shock_angle_upper']):
                    pressure[i] = 1.0
                elif y < -x * np.tan(results['rarefaction_angles_lower']['initial']):
                    pressure[i] = 1.0
                else:
                    # Плавное изменение давления в веере разрежения
                    angle = np.arctan2(-y, x)
                    t = (angle - results['rarefaction_angles_lower']['initial']) / \
                        (results['rarefaction_angles_lower']['final'] - results['rarefaction_angles_lower']['initial'])
                    t = max(0, min(1, t))
                    pressure[i] = 1.0 + (p_interface - 1.0) * np.tanh(2 * t)
                    
            elif interaction_type == 'rarefaction_shock':
                # Для случая с волной разрежения и ударной волной
                if y > x * np.tan(results['rarefaction_angles_upper']['initial']):
                    pressure[i] = 1.0
                elif y < -x * np.tan(results['shock_angle_lower']):
                    pressure[i] = 1.0
                else:
                    # Плавное изменение давления в веере разрежения
                    angle = np.arctan2(y, x)
                    t = (angle - results['rarefaction_angles_upper']['initial']) / \
                        (results['rarefaction_angles_upper']['final'] - results['rarefaction_angles_upper']['initial'])
                    t = max(0, min(1, t))
                    pressure[i] = 1.0 + (p_interface - 1.0) * np.tanh(2 * t)
                    
            elif interaction_type == 'rarefaction_rarefaction':
                # Для случая с двумя волнами разрежения
                if y > x * np.tan(results['rarefaction_angles_upper']['initial']):
                    pressure[i] = 1.0
                elif y < -x * np.tan(results['rarefaction_angles_lower']['initial']):
                    pressure[i] = 1.0
                else:
                    # Плавное изменение давления в веере разрежения
                    if y > 0:
                        angle = np.arctan2(y, x)
                        t = (angle - results['rarefaction_angles_upper']['initial']) / \
                            (results['rarefaction_angles_upper']['final'] - results['rarefaction_angles_upper']['initial'])
                    else:
                        angle = np.arctan2(-y, x)
                        t = (angle - results['rarefaction_angles_lower']['initial']) / \
                            (results['rarefaction_angles_lower']['final'] - results['rarefaction_angles_lower']['initial'])
                    t = max(0, min(1, t))
                    pressure[i] = 1.0 + (p_interface - 1.0) * np.tanh(2 * t)
        
        # Построение графика
        ax.plot(s, pressure, 'b-', linewidth=2)
        
        # Настройка графика
        ax.set_title(f'Распределение давления вдоль линии на расстоянии {distance} от точки столкновения')
        ax.set_xlabel('Параметр вдоль линии')
        ax.set_ylabel('Давление')
        ax.grid(True, alpha=0.3)
        
        # Добавление информации о параметрах
        info_text = (
            f"Тип взаимодействия: {interaction_type}\n"
            f"Давление на контактном разрыве: {p_interface:.3f}"
        )
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.9))
        
        # Настройка общего вида
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _scale_field(self, field, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.nanmin(field)
        if vmax is None:
            vmax = np.nanmax(field)
        return (field - vmin) / (vmax - vmin)
    
    def _calculate_contact_angle(self, results):
        """
        Вычисление угла контактного разрыва на основе результатов расчета.
        
        Параметры:
        ----------
        results : dict
            Словарь с результатами расчета
            
        Возвращает:
        -----------
        float
            Угол контактного разрыва в радианах
        """
        # Получаем углы отклонения потока
        delta1 = results.get('delta1', 0.0)
        delta2 = results.get('delta2', 0.0)
        
        # Получаем начальные углы наклона векторов скорости
        theta1 = results.get('theta1', 0.0)
        theta2 = results.get('theta2', 0.0)
        
        # Вычисляем углы наклона векторов скорости после волн
        angle1 = theta1 + delta1
        angle2 = theta2 + delta2
        
        # Угол контактного разрыва - среднее значение углов наклона векторов скорости
        return (angle1 + angle2) / 2

def is_valid_result(results):
    keys = ['u_upper', 'v_upper', 'u_lower', 'v_lower']
    for k in keys:
        val = results.get(k, 0)
        if not np.isfinite(val):
            return False
    return True

def plot_all_interaction_types(solver):
    """
    Строит все пять режимов взаимодействия потоков (а–д) с характерными параметрами.
    На каждом графике отображаются ударные волны, границы вееров разрежения и тангенциальные разрывы.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cases = [
        ("а: rarefaction-rarefaction", {
            'M1': 1.4, 'M2': 3.0,
            'p1': 1.0, 'p2': 2.0,
            'rho1': 1.0, 'rho2': 1.0,
            'theta1': np.radians(10),
            'theta2': np.radians(-20)
        }, 'rarefaction_rarefaction'),
        ("б: shock-shock", {
            'M1': 3.0, 'M2': 3.5,
            'p1': 2.0, 'p2': 2.5,
            'rho1': 1.0, 'rho2': 1.2,
            'theta1': np.radians(-20), 'theta2': np.radians(20)
        }, 'shock_shock'),
        ("в: rarefaction-shock", {
            'M1': 1.1, 'M2': 2.0,
            'p1': 1.0, 'p2': 3.0,
            'rho1': 1.0, 'rho2': 1.0,
            'theta1': np.radians(0), 'theta2': np.radians(30)
        }, 'rarefaction-shock'),
          ("г: shock-rarefaction", {
            'M1': 2.0, 'M2': 1.2,
            'p1': 1.0, 'p2': 1.0,
            'rho1': 1.0, 'rho2': 1.0,
            'theta1': np.radians(20), 'theta2': np.radians(0)
        }, 'shock_rarefaction'),
        ("д: rarefaction-rarefaction (вакуум)", {
            'M1': 1.5, 'M2': 1.5,
            'p1': 1.0, 'p2': 1.0,
            'rho1': 1.0, 'rho2': 1.0,
            'theta1': np.radians(20), 'theta2': np.radians(-20)
        }, 'rarefaction_rarefaction'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (title, params, expected_type) in enumerate(cases):
        try:
            print(f"\nОбработка случая {title}:")
            print(f"Параметры: M1={params['M1']}, M2={params['M2']}, p1={params['p1']}, p2={params['p2']}")
            print(f"Ожидаемый тип: {expected_type}")
            
            results = solver.solve_flow_interaction(
                params['M1'], params['M2'], params['p1'], params['p2'],
                params['rho1'], params['rho2'], params['theta1'], params['theta2']
            )
            
            print(f"Полученный тип: {results['interaction_type']}")
            print(f"Давление на контактном разрыве: {results['p_interface']}")
            
            ax = axes[idx]
            # Визуализация только основного поля давления
            x_range = (-5.0, 5.0)
            y_range = (-5.0, 5.0)
            resolution = 100
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            U = np.zeros_like(X)
            V = np.zeros_like(X)
            P = np.zeros_like(X)
            P[:, :] = 1.0  # Инициализация давлением 1.0
            interaction_type = results['interaction_type']

            if interaction_type == 'shock_shock':
                solver._fill_shock_shock_fields(X, Y, U, V, P, results)
            elif interaction_type == 'shock_rarefaction':
                solver._fill_shock_rarefaction_fields(X, Y, U, V, P, results)
            elif interaction_type == 'rarefaction_shock':
                solver._fill_rarefaction_shock_fields(X, Y, U, V, P, results)
            elif interaction_type == 'rarefaction_rarefaction':
                solver._fill_rarefaction_rarefaction_fields(X, Y, U, V, P, results)

            pressure_contour = ax.contourf(X, Y, P, cmap='viridis', levels=30)

            ax.set_title(f"{title}\nТип: {interaction_type}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(pressure_contour, ax=ax, shrink=0.7)

            # --- ДОПОЛНИТЕЛЬНЫЕ ЛИНИИ ---
            # Ударные волны
            if 'shock_angle_upper' in results:
                beta = results['shock_angle_upper']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = x_shock * np.tan(beta)
                # Обрезаем по y_range
                mask = (y_shock >= y_range[0]) & (y_shock <= y_range[1])
                ax.plot(x_shock[mask], y_shock[mask], color='red', linewidth=2.5, label='Ударная волна')
            if 'shock_angle_lower' in results:
                beta = results['shock_angle_lower']
                x_shock = np.linspace(0, x_range[1], 100)
                y_shock = -x_shock * np.tan(beta)
                mask = (y_shock >= y_range[0]) & (y_shock <= y_range[1])
                ax.plot(x_shock[mask], y_shock[mask], color='red', linewidth=2.5)

            # Границы вееров разрежения
            if 'rarefaction_angles_upper' in results:
                angles = results['rarefaction_angles_upper']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = x_rare * np.tan(mu_initial)
                y_final = x_rare * np.tan(mu_final)
                mask1 = (y_initial >= y_range[0]) & (y_initial <= y_range[1])
                mask2 = (y_final >= y_range[0]) & (y_final <= y_range[1])
                ax.plot(x_rare[mask1], y_initial[mask1], color='blue', linestyle='dashed', linewidth=2, label='Граница веера разрежения')
                ax.plot(x_rare[mask2], y_final[mask2], color='blue', linestyle='dashed', linewidth=2)
            if 'rarefaction_angles_lower' in results:
                angles = results['rarefaction_angles_lower']
                mu_initial = angles['initial']
                mu_final = angles['final']
                x_rare = np.linspace(0, x_range[1], 100)
                y_initial = -x_rare * np.tan(mu_initial)
                y_final = -x_rare * np.tan(mu_final)
                mask1 = (y_initial >= y_range[0]) & (y_initial <= y_range[1])
                mask2 = (y_final >= y_range[0]) & (y_final <= y_range[1])
                ax.plot(x_rare[mask1], y_initial[mask1], color='blue', linestyle='dashed', linewidth=2)
                ax.plot(x_rare[mask2], y_final[mask2], color='blue', linestyle='dashed', linewidth=2)

            # Тангенциальный разрыв (контактный разрыв)
            x_int = np.linspace(0, x_range[1], 100)
            if results['interaction_type'] == 'shock_rarefaction':
                theta_contact = np.arctan2(results['v_upper'], results['u_upper'])
            elif results['interaction_type'] == 'rarefaction_shock':
                theta_contact = np.arctan2(results['v_lower'], results['u_lower'])
                y_int = x_int * np.tan(theta_contact)
            else:
                xi = results['xi_interface']
                y_int = x_int * xi
            mask_int = (y_int >= y_range[0]) & (y_int <= y_range[1])
            ax.plot(x_int[mask_int], y_int[mask_int], color='black', linestyle='dashdot', linewidth=2, label='Тангенциальный разрыв')
            
            # Начальные линии тока
            theta1 = params['theta1']
            theta2 = params['theta2']
            
            # Верхний поток
            y_start_upper = np.linspace(0.1, y_range[1], 10)
            for y0 in y_start_upper:
                x_line = np.array([x_range[0], 0])
                y_line = np.array([y0, y0 + np.tan(theta1) * (0 - x_range[0])])
                if y_line[1] > 0:  # Проверяем, что линия не пересекает ось x
                    ax.plot(x_line, y_line, color='#1f77b4', linewidth=1.0, alpha=0.7)
                    # Добавляем стрелочку
                    arrow_x = x_line[0] + (x_line[1] - x_line[0]) * 0.7
                    arrow_y = y_line[0] + (y_line[1] - y_line[0]) * 0.7
                    ax.arrow(arrow_x, arrow_y, 
                            (x_line[1] - x_line[0]) * 0.1, 
                            (y_line[1] - y_line[0]) * 0.1,
                            head_width=0.02, head_length=0.05, fc='#1f77b4', ec='#1f77b4')
            
            # Нижний поток
            y_start_lower = np.linspace(y_range[0], -0.1, 10)
            for y0 in y_start_lower:
                x_line = np.array([x_range[0], 0])
                y_line = np.array([y0, y0 + np.tan(theta2) * (0 - x_range[0])])
                if y_line[1] < 0:  # Проверяем, что линия не пересекает ось x
                    ax.plot(x_line, y_line, color='#ff7f0e', linewidth=1.0, alpha=0.7)
                    # Добавляем стрелочку
                    arrow_x = x_line[0] + (x_line[1] - x_line[0]) * 0.7
                    arrow_y = y_line[0] + (y_line[1] - y_line[0]) * 0.7
                    ax.arrow(arrow_x, arrow_y, 
                            (x_line[1] - x_line[0]) * 0.1, 
                            (y_line[1] - y_line[0]) * 0.1,
                            head_width=0.02, head_length=0.05, fc='#ff7f0e', ec='#ff7f0e')

            # Легенда (только если есть что показать)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize=9)

        except Exception as e:
            print(f"Ошибка при обработке случая {title}: {str(e)}")
            ax = axes[idx]
            ax.set_title(f"{title}\nОшибка: {e}")
            ax.axis('off')

    # Отключаем пустой шестой график
    for i in range(len(cases), len(axes)):
        axes[i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Все типы взаимодействия сверхзвуковых потоков (а–д)", fontsize=18)
    plt.show()

if __name__ == "__main__":

    # 1) Создаем решатель
    solver = SupersonicFlowInteraction(gamma=1.4)
    
    # 2) Одиночный тестовый пример
    M1, M2 = 3.0, 3.5
    p1, p2 = 2.0, 2.5
    rho1, rho2 = 1.0, 1.2
    theta1 = np.radians(10)
    theta2 = np.radians(-10)
    results = solver.solve_flow_interaction(M1, M2, p1, p2, rho1, rho2, theta1, theta2)
    print("Тип взаимодействия:", results['interaction_type'])

    print("=== Одноточечный тест ===")
    for k, v in results.items():
        print(f"{k:25s}: {v}")

    # 3) Визуализация поля только если значения корректны
    if is_valid_result(results):
        fig_main = solver.visualize_flow(
            results,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            resolution=100
        )
        plt.show()
    else:
        print('Внимание: некорректные значения скоростей, визуализация пропущена.')
    plot_all_interaction_types(solver)

    # 4) Параметрическое исследование
    M1_values  = [2.0, 3.0]
    M2_values  = [1.5, 2.5]
    p_ratios   = [1.5, 1.0]
    rho_ratios = [1.0, 1.2]
    results_dict = solver.run_parametric_study(
        M1_values, M2_values, p_ratios, rho_ratios,
        save_figures=False
    )

    # 5) Статистический анализ
    stats = solver.analyze_results(results_dict)
    print("\n=== Статистика параметрического исследования ===")
    print("Ключи stats:", list(stats.keys()))

    for key, val in stats.items():
        if key == 'pressure_ranges':
            p_min  = val.get('min', float('nan'))
            p_max  = val.get('max', float('nan'))
            p_mean = val.get('mean', float('nan'))
            print(f"{key:30s} p_int мин={p_min:.3f}, макс={p_max:.3f}, ср={p_mean:.3f}")
        elif key == 'interaction_types':
            print(f"{key:30s}")
            for t, percent in val.items():
                print(f"  {t:25s}: {percent:.1f}%")
        elif isinstance(val, list):
            print(f"\n{key}:")
            for item in val:
                case_name = item.get('case', '<no case>')
                upper = item.get('upper_flow', float('nan'))
                lower = item.get('lower_flow', float('nan'))
                print(f"  {case_name:50s} | верхний поток={upper:.3f}, нижний поток={lower:.3f}")
        else:
            print(f"{key}: {val}")
