/**
 * Scroll to Top Button - UX Enhancement
 */

document.addEventListener('DOMContentLoaded', function() {
    // Criar botão scroll to top
    const scrollBtn = document.createElement('button');
    scrollBtn.id = 'scroll-to-top';
    scrollBtn.innerHTML = '<i class="bi bi-arrow-up"></i>';
    scrollBtn.setAttribute('title', 'Voltar ao topo');
    scrollBtn.setAttribute('aria-label', 'Voltar ao topo da página');
    document.body.appendChild(scrollBtn);
    
    // Mostrar/ocultar botão baseado no scroll
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            scrollBtn.classList.add('show');
        } else {
            scrollBtn.classList.remove('show');
        }
    });
    
    // Scroll suave ao topo ao clicar
    scrollBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    console.log('✅ Scroll to top button inicializado');
});

/**
 * Keyboard Shortcuts
 */
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K: Foco na barra de busca (se existir)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Home: Scroll to top
    if (e.key === 'Home' && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // End: Scroll to bottom
    if (e.key === 'End' && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
});

/**
 * Smooth scroll para links âncora
 */
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const targetId = e.target.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

/**
 * Loading feedback visual
 */
window.addEventListener('load', function() {
    // Remover skeleton loaders
    const skeletons = document.querySelectorAll('.skeleton');
    skeletons.forEach(function(skeleton) {
        skeleton.classList.add('loaded');
        setTimeout(function() {
            skeleton.style.display = 'none';
        }, 300);
    });
    
    console.log('✅ Página carregada completamente');
});
