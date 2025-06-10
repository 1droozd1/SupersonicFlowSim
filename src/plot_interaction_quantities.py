import numpy as np
import matplotlib.pyplot as plt
from main import SupersonicFlowInteraction
from scipy.optimize import fsolve

def prandtl_meyer_nu(M, gamma):
    """
    Функция Прандтля-Майера для вычисления угла поворота потока.
    
    Параметры:
    ----------
    M : float
        Число Маха
    gamma : float
        Показатель адиабаты
    
    Возвращает:
    -----------
    float
        Угол поворота потока в радианах
    """
    # Проверка на M < 1.0, для подкритических чисел Маха функция не определена
    if M < 1.0:
        return 0.0
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))

def prandtl_meyer_inverse(nu, gamma):
    def func(M):
        return prandtl_meyer_nu(M, gamma) - nu
    M_guess = 2.0 if nu > 0 else 1.01
    M_solution = fsolve(func, M_guess)[0]
    return M_solution

def plot_interaction_quantities(solver, results, radius=1.0, num_points=2000):
    """
    Построение графиков изменения величин взаимодействия на фиксированном радиусе.
    
    Параметры:
    ----------
    solver : SupersonicFlowInteraction
        Экземпляр решателя
    results : dict
        Результаты расчета
    radius : float
        Радиус, на котором строятся графики
    num_points : int
        Количество точек для построения
    """
    # Добавляем расчет M_interface_upper и M_interface_lower, если их нет в results
    if 'M_interface_upper' not in results:
        c_interface_upper = np.sqrt(solver.gamma * results['p_interface'] / results['rho_upper'])
        results['M_interface_upper'] = np.sqrt(results['u_upper']**2 + results['v_upper']**2) / c_interface_upper
    if 'M_interface_lower' not in results:
        c_interface_lower = np.sqrt(solver.gamma * results['p_interface'] / results['rho_lower'])
        results['M_interface_lower'] = np.sqrt(results['u_lower']**2 + results['v_lower']**2) / c_interface_lower
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Определяем диапазон углов в зависимости от типа взаимодействия
    interaction_type = results['interaction_type']
    
    margin = np.radians(40)  # запас для визуализации

    if interaction_type == 'shock_shock':
        angle_min = -results['shock_angle_lower'] - margin
        angle_max = results['shock_angle_upper'] + margin
    elif interaction_type == 'shock_rarefaction':
        angle_min = results['rarefaction_angles_lower']['initial'] + margin
        angle_max = -results['shock_angle_upper'] - margin
    elif interaction_type == 'rarefaction_shock':
        angle_min = results['shock_angle_lower'] + margin
        angle_max = results['rarefaction_angles_upper']['final'] - margin
    else:  # rarefaction_rarefaction
        rarefaction_angle_upper_initial = results['rarefaction_angles_upper']['initial']
        rarefaction_angle_upper_final = results['rarefaction_angles_upper']['final']
        rarefaction_angle_lower_initial = -results['rarefaction_angles_lower']['initial']
        rarefaction_angle_lower_final = -results['rarefaction_angles_lower']['final']

        angle_min = min(
            rarefaction_angle_upper_initial,
            rarefaction_angle_lower_initial,
        ) - margin
        angle_max = max(
            rarefaction_angle_upper_final,
            rarefaction_angle_lower_final
        ) + margin
    # Создаем массив углов с расширенным диапазоном
    angles = np.linspace(angle_min, angle_max, num_points)
    
    # Создаем массивы для хранения величин
    pressure = np.zeros_like(angles)
    density = np.zeros_like(angles)
    velocity_magnitude = np.zeros_like(angles)
    mach_number = np.zeros_like(angles)
    
    # Определяем угол контактного разрыва
    contact_angle = solver._calculate_contact_angle(results)
    has_contact = True
    
    # Определяем регионы в зависимости от типа взаимодействия
    if interaction_type == 'shock_shock':
        # Регионы для случая с двумя ударными волнами
        shock_angle_upper = results['shock_angle_upper']
        shock_angle_lower = results['shock_angle_lower']
        
        # Регион 1: до верхней ударной волны
        mask1 = angles > shock_angle_upper
        pressure[mask1] = results['p1']
        density[mask1] = results['rho1']
        c1 = np.sqrt(solver.gamma * results['p1'] / results['rho1'])
        velocity_magnitude[mask1] = results['M1'] * c1
        
        # Регион 2: между верхней ударной волной и контактным разрывом
        mask2 = (angles <= shock_angle_upper) & (angles > contact_angle)
        pressure[mask2] = results['p_interface']
        density[mask2] = results['rho_upper']
        velocity_magnitude[mask2] = np.sqrt(results['u_upper']**2 + results['v_upper']**2)
        
        # Регион 3: между контактным разрывом и нижней ударной волной
        mask3 = (angles <= contact_angle) & (angles > -shock_angle_lower)
        pressure[mask3] = results['p_interface']
        density[mask3] = results['rho_lower']
        velocity_magnitude[mask3] = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
        
        # Регион 4: после нижней ударной волны
        mask4 = angles <= -shock_angle_lower
        pressure[mask4] = results['p2']
        density[mask4] = results['rho2']
        c2 = np.sqrt(solver.gamma * results['p2'] / results['rho2'])
        velocity_magnitude[mask4] = results['M2'] * c2
        
    elif interaction_type == 'shock_rarefaction':
        # Регионы для случая с ударной волной и волной разрежения
        shock_angle_upper = -results['shock_angle_upper']
        rarefaction_angle_lower_initial = results['rarefaction_angles_lower']['initial']
        rarefaction_angle_lower_final = results['rarefaction_angles_lower']['final']
        print(np.degrees(rarefaction_angle_lower_initial), np.degrees(rarefaction_angle_lower_final), np.degrees(shock_angle_upper), contact_angle)
        
        # Регион 1: до ударной волны
        mask1 = angles < shock_angle_upper
        pressure[mask1] = results['p1']
        density[mask1] = results['rho1']
        c1 = np.sqrt(solver.gamma * results['p1'] / results['rho1'])
        velocity_magnitude[mask1] = results['M1'] * c1
        
        # Регион 2: между ударной волной и контактным разрывом
        mask2 = (angles >= shock_angle_upper) & (angles < contact_angle)
        pressure[mask2] = results['p_interface']
        density[mask2] = results['rho_upper']
        velocity_magnitude[mask2] = np.sqrt(results['u_upper']**2 + results['v_upper']**2)
        
        # Регион 3: между контактным разрывом и началом веера разрежения
        mask3 = (angles >= contact_angle) & (angles < rarefaction_angle_lower_initial)
        pressure[mask3] = results['p_interface']
        density[mask3] = results['rho_lower']
        velocity_magnitude[mask3] = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
        
        # Регион 4: в веере разрежения
        mask4 = (angles >= rarefaction_angle_lower_initial) & (angles <= rarefaction_angle_lower_final)
        if np.any(mask4):
            # Значения на границах веера
            p_max = results['p_interface']
            p_min = results['p2']
            rho_max = results['rho_lower']
            rho_min = results['rho2']
            v_max = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
            v_min = results['M2'] * np.sqrt(solver.gamma * results['p2'] / results['rho2'])
            t = (angles[mask4] - rarefaction_angle_lower_initial) / (rarefaction_angle_lower_final - rarefaction_angle_lower_initial)
            t = np.clip(t, 0.0, 1.0)
            pressure[mask4] = p_max * (1 - t) + p_min * t
            density[mask4] = rho_max * (1 - t) + rho_min * t
            velocity_magnitude[mask4] = v_max * (1 - t) + v_min * t
        
        # Регион 5: после веера разрежения
        mask5 = angles >= rarefaction_angle_lower_final
        pressure[mask5] = results['p2']
        density[mask5] = results['rho2']
        c2 = np.sqrt(solver.gamma * results['p2'] / results['rho2'])
        velocity_magnitude[mask5] = results['M2'] * c2
        
    elif interaction_type == 'rarefaction_shock':
        # Регионы для случая с волной разрежения и ударной волной
        rarefaction_angle_upper_initial = results['rarefaction_angles_upper']['initial']
        rarefaction_angle_upper_final = results['rarefaction_angles_upper']['final']
        shock_angle_lower = results['shock_angle_lower']
        
        # Регион 1: до начала веера разрежения
        mask1 = angles <= rarefaction_angle_upper_initial
        pressure[mask1] = results['p1']
        density[mask1] = results['rho1']
        c1 = np.sqrt(solver.gamma * results['p1'] / results['rho1'])
        velocity_magnitude[mask1] = results['M1'] * c1
        
        # Регион 2: в веере разрежения
        mask2 = (angles >= rarefaction_angle_upper_initial) & (angles <= rarefaction_angle_upper_final)
        if np.any(mask2):
            # Значения на границах веера
            p_max = results['p1']  # давление до веера (исходное)
            p_min = results['p_interface']  # давление после веера (на контактном разрыве)
            rho_max = results['rho1']       # плотность до веера (исходная)
            rho_min = results['rho_upper']  # плотность после веера (за веером)
            v_max = results['M1'] * np.sqrt(solver.gamma * results['p1'] / results['rho1'])  # скорость до веера
            v_min = np.sqrt(results['u_upper']**2 + results['v_upper']**2)                   # скорость после веера

            t = (angles[mask2] - rarefaction_angle_upper_initial) / (rarefaction_angle_upper_final - rarefaction_angle_upper_initial)
            t = np.clip(t, 0.0, 1.0)
            pressure[mask2] = (p_min * (1 - t) + p_max * t)[::-1]
            density[mask2] = (rho_max * (1 - t) + rho_min * t)[::]
            velocity_magnitude[mask2] = (v_max * (1 - t) + v_min * t)[::]
        
        # Регион 3: между концом веера разрежения и контактным разрывом
        mask3 = (angles >= rarefaction_angle_upper_final) & (angles < contact_angle)
        pressure[mask3] = results['p_interface']
        density[mask3] = results['rho_upper']
        velocity_magnitude[mask3] = np.sqrt(results['u_upper']**2 + results['v_upper']**2)
        
        # Регион 4: между контактным разрывом и ударной волной
        mask4 = (angles >= contact_angle) & (angles < shock_angle_lower)
        pressure[mask4] = results['p_interface']
        density[mask4] = results['rho_lower']
        velocity_magnitude[mask4] = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
        
        # Регион 5: после ударной волны
        mask5 = angles >= shock_angle_lower
        pressure[mask5] = results['p2']
        density[mask5] = results['rho2']
        c2 = np.sqrt(solver.gamma * results['p2'] / results['rho2'])
        velocity_magnitude[mask5] = results['M2'] * c2
        
    else:  # rarefaction_rarefaction
        # Регионы для случая с двумя волнами разрежения
        rarefaction_angle_upper_initial = results['rarefaction_angles_upper']['initial']
        rarefaction_angle_upper_final = results['rarefaction_angles_upper']['final']
        rarefaction_angle_lower_initial = -results['rarefaction_angles_lower']['initial']
        rarefaction_angle_lower_final = -results['rarefaction_angles_lower']['final']
        
        # Регион 1: до начала верхнего веера разрежения
        mask1 = angles > rarefaction_angle_upper_initial
        pressure[mask1] = results['p1']
        density[mask1] = results['rho1']
        c1 = np.sqrt(solver.gamma * results['p1'] / results['rho1'])
        velocity_magnitude[mask1] = results['M1'] * c1
        
        # Регион 2: в верхнем веере разрежения
        mask2 = (angles >= rarefaction_angle_upper_initial) & (angles <= rarefaction_angle_upper_final)
        if np.any(mask2):
            # Значения на границах веера
            p_max = results['p1']  # давление до веера (исходное)
            p_min = results['p_interface']  # давление после веера (на контактном разрыве)
            rho_max = results['rho1']       # плотность до веера (исходная)
            rho_min = results['rho_upper']  # плотность после веера (за веером)
            v_max = results['M1'] * np.sqrt(solver.gamma * results['p1'] / results['rho1'])  # скорость до веера
            v_min = np.sqrt(results['u_upper']**2 + results['v_upper']**2)                   # скорость после веера

            t = (angles[mask2] - rarefaction_angle_upper_initial) / (rarefaction_angle_upper_final - rarefaction_angle_upper_initial)
            t = np.clip(t, 0.0, 1.0)
            pressure[mask2] = (p_min * (1 - t) + p_max * t)
            density[mask2] = (rho_max * (1 - t) + rho_min * t)[::-1]
            velocity_magnitude[mask2] = (v_max * (1 - t) + v_min * t)[::-1]
        
        # Регион 3: между концом верхнего веера и контактным разрывом
        mask3 = (angles <= rarefaction_angle_upper_initial) & (angles > contact_angle)
        print(np.sum(mask2 & mask3))
        pressure[mask3] = results['p_interface']
        density[mask3] = results['rho_upper']
        velocity_magnitude[mask3] = np.sqrt(results['u_upper']**2 + results['v_upper']**2)
        
        # Регион 4: между контактным разрывом и началом нижнего веера
        mask4 = (angles <= contact_angle) & (angles > rarefaction_angle_lower_initial)
        pressure[mask4] = results['p_interface']
        density[mask4] = results['rho_lower']
        velocity_magnitude[mask4] = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
        
        # Регион 5: в нижнем веере разрежения
        mask5 = (angles <= rarefaction_angle_lower_initial) & (angles > rarefaction_angle_lower_final)
        
        if np.any(mask5):
            # Значения на границах веера
            p_max = results['p_interface']
            p_min = results['p2']
            rho_max = results['rho_lower']
            rho_min = results['rho2']
            v_max = np.sqrt(results['u_lower']**2 + results['v_lower']**2)
            v_min = results['M2'] * np.sqrt(solver.gamma * results['p2'] / results['rho2'])
            t = (angles[mask5] - rarefaction_angle_lower_initial) / (rarefaction_angle_lower_final - rarefaction_angle_lower_initial)
            t = np.clip(t, 0.0, 1.0)
            pressure[mask5] = p_max * (1 - t) + p_min * t
            density[mask5] = rho_max * (1 - t) + rho_min * t
            velocity_magnitude[mask5] = v_max * (1 - t) + v_min * t
        
        # Регион 6: после нижнего веера разрежения
        mask6 = angles <= rarefaction_angle_lower_final
        pressure[mask6] = results['p2']
        density[mask6] = results['rho2']
        c2 = np.sqrt(solver.gamma * results['p2'] / results['rho2'])
        velocity_magnitude[mask6] = results['M2'] * c2
    
    # Вычисляем число Маха
    speed_of_sound_sq = solver.gamma * pressure / density
    mach_number = np.where(speed_of_sound_sq > 1e-9, velocity_magnitude / np.sqrt(speed_of_sound_sq), 0.0)
    
    # Построение графиков
    # 1. Давление
    ax = axes[0]
    ax.plot(np.degrees(angles), pressure, 'b-', linewidth=2)
    ax.set_title('Распределение давления')
    ax.set_xlabel('Угол (градусы)')
    ax.set_ylabel('Давление')
    ax.grid(True)
    
    # Добавляем вертикальные линии для ударных волн, границ вееров и контактного разрыва
    if interaction_type == 'shock_shock':
        ax.axvline(np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5, label='Ударная волна')
        ax.axvline(-np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5, label='Контактный разрыв')
    elif interaction_type == 'shock_rarefaction':
        ax.axvline(-np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5, label='Ударная волна')
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5, label='Граница веера')
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5, label='Контактный разрыв')
    elif interaction_type == 'rarefaction_shock':
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5, label='Граница веера')
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5, label='Ударная волна')
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5, label='Контактный разрыв')
    else:  # rarefaction_rarefaction
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5, label='Граница веера')
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5, label='Контактный разрыв')
    
    ax.legend()
    
    # 2. Плотность
    ax = axes[1]
    ax.plot(np.degrees(angles), density, 'r-', linewidth=2)
    ax.set_title('Распределение плотности')
    ax.set_xlabel('Угол (градусы)')
    ax.set_ylabel('Плотность')
    ax.grid(True)
    
    # Добавляем вертикальные линии для всех типов взаимодействий на графике плотности
    if interaction_type == 'shock_shock':
        ax.axvline(np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'shock_rarefaction':
        ax.axvline(-np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'rarefaction_shock':
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    else:  # rarefaction_rarefaction
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    
    # 3. Модуль скорости
    ax = axes[2]
    ax.plot(np.degrees(angles), velocity_magnitude, 'g-', linewidth=2)
    ax.set_title('Распределение модуля скорости')
    ax.set_xlabel('Угол (градусы)')
    ax.set_ylabel('|V|')
    ax.grid(True)
    
    # Добавляем вертикальные линии для всех типов взаимодействий на графике модуля скорости
    if interaction_type == 'shock_shock':
        ax.axvline(np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'shock_rarefaction':
        ax.axvline(-np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'rarefaction_shock':
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    else:  # rarefaction_rarefaction
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    
    # 4. Число Маха
    ax = axes[3]
    ax.plot(np.degrees(angles), mach_number, 'm-', linewidth=2)
    ax.set_title('Распределение числа Маха')
    ax.set_xlabel('Угол (градусы)')
    ax.set_ylabel('M')
    ax.grid(True)
    
    # Добавляем вертикальные линии для всех типов взаимодействий на графике числа Маха
    if interaction_type == 'shock_shock':
        ax.axvline(np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'shock_rarefaction':
        ax.axvline(-np.degrees(results['shock_angle_upper']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    elif interaction_type == 'rarefaction_shock':
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['shock_angle_lower']), color='r', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    else:  # rarefaction_rarefaction
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(results['rarefaction_angles_upper']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['initial']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(-np.degrees(results['rarefaction_angles_lower']['final']), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(contact_angle), color='g', linestyle='-', alpha=0.5)
    
    # Добавление информации о параметрах
    info_text = (
        f"Тип взаимодействия: {interaction_type}\n"
        f"Радиус: {radius}\n"
        f"Давление на контактном разрыве: {results['p_interface']:.3f}\n"
        f"Плотности: ρ₁ = {results['rho_upper']:.3f}, ρ₂ = {results['rho_lower']:.3f}\n"
        f"Скорости: |V₁| = {np.sqrt(results['u_upper']**2 + results['v_upper']**2):.3f}, "
        f"|V₂| = {np.sqrt(results['u_lower']**2 + results['v_lower']**2):.3f}\n"
        f"\nВходные параметры:\n"
        f"M₁ = {results['M1']:.2f}, M₂ = {results['M2']:.2f}\n"
        f"p₁ = {results['p1']:.2f}, p₂ = {results['p2']:.2f}\n"
        f"ρ₁ = {results['rho1']:.2f}, ρ₂ = {results['rho2']:.2f}\n"
        f"θ₁ = {np.degrees(results['theta1']):.1f}°, θ₂ = {np.degrees(results['theta2']):.1f}°"
    )
    if has_contact:
        info_text += f"\nУгол контактного разрыва: {np.degrees(contact_angle):.1f}°"
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.9))
    
    # Настройка общего вида
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Распределение величин на радиусе {radius}", fontsize=16)
    
    return fig

if __name__ == "__main__":
    # Создаем решатель
    solver = SupersonicFlowInteraction(gamma=1.4)
    
    # Тестовый случай
    params = {
        'M1': 3.0, 'M2': 3.5,
        'p1': 2.0, 'p2': 2.5,
        'rho1': 1.0, 'rho2': 1.2,
        'theta1': np.radians(20),
        'theta2': np.radians(-20)
    }
    
    # Решение задачи
    results = solver.solve_flow_interaction(
        params['M1'], params['M2'],
        params['p1'], params['p2'],
        params['rho1'], params['rho2'],
        params['theta1'], params['theta2']
    )
    
    # Вывод информации о типе взаимодействия
    print("\nИнформация о взаимодействии:")
    print(f"Тип взаимодействия: {results['interaction_type']}")
    print(f"Давление на контактном разрыве: {results['p_interface']:.3f}")
    print(f"Начальные давления: p1 = {params['p1']:.3f}, p2 = {params['p2']:.3f}")
    print(f"Углы отклонения потока: delta1 = {np.degrees(results['delta1']):.1f}°, delta2 = {np.degrees(results['delta2']):.1f}°")
    print(f"Плотности после волн: rho_upper = {results['rho_upper']:.3f}, rho_lower = {results['rho_lower']:.3f}")
    print(f"Скорости после волн: |V1| = {np.sqrt(results['u_upper']**2 + results['v_upper']**2):.3f}, |V2| = {np.sqrt(results['u_lower']**2 + results['v_lower']**2):.3f}")
    
    # Построение графиков
    fig = plot_interaction_quantities(solver, results, radius=1.0)
    plt.savefig(f"results/interaction_quantities_{results['interaction_type']}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)