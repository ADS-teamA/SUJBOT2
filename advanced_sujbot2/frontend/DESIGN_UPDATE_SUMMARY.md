# Modern Black & White Design Update

## Overview
Aplikace byla modernizována s novým minimalistickým černobílým designem, včetně gradientů a plynulých animací.

## Provedené změny

### 1. **Barevná paleta** (`src/index.css`)
- **Light mode**: Čistě bílá (`100%`) s jemnými odstíny šedi (`98%`, `96%`, `94%`)
- **Dark mode**: Černá (`8%`) s tmavými odstíny šedi (`12%`, `16%`, `18%`)
- Primární barvy nastaveny na černou/bílou místo barevných tónů
- Radius zvětšen na `0.75rem` pro modernější vzhled

### 2. **Gradienty** (`src/index.css` + `tailwind.config.js`)

#### Custom CSS utility třídy:
- `.gradient-subtle`: Subtilní gradient pro pozadí
- `.gradient-border`: Gradient text efekt
- `.transition-smooth`: Plynulé přechody (cubic-bezier)
- `.transition-bounce`: Bounce efekt pro interaktivní prvky

#### Tailwind animace:
- `animate-fade-in`: Objevení prvku s fade in efektem
- `animate-fade-in-up`: Fade in s pohybem nahoru
- `animate-slide-in-right/left`: Slide-in animace
- `animate-scale-in`: Scale-in efekt
- `animate-shimmer`: Shimmer efekt pro loading stavy
- `animate-pulse-slow`: Pomalý pulse pro pozadí

### 3. **Komponenty**

#### **TopBar** (`components/TopBar.tsx`)
- Gradient pozadí s přechodem `from-white via-gray-50 to-white`
- Logo s glow efektem při hover
- Gradient text pro název aplikace
- Smooth scale efekt na ikoně při hover
- Fade-in animace při načtení

#### **App** (`App.tsx`)
- Gradient pozadí celé aplikace
- Jemné oddělovače mezi panely pomocí gradientních pruhů
- Smooth transitions mezi light/dark módem

#### **ChatArea** (`components/ChatArea.tsx`)
- Gradient pozadí chat oblasti
- Welcome screen s animovaným názvem a glow efektem
- Každá zpráva se objeví s fade-in animací
- Delay mezi zprávami pro plynulejší efekt

#### **DocumentPanel** (`components/DocumentPanel.tsx`)
- Vertikální gradient pozadí
- Gradient text v nadpisu
- Animace dokumentů při načtení s delay

#### **Button** (`components/ui/button.tsx`)
- Hover efekt s scale (`scale-105`)
- Shadow efekty při hover
- Smooth transitions (300ms)
- Group utility pro nested animace

#### **Card** (`components/ui/card.tsx`)
- Shadow upgrade (`shadow-lg` → `shadow-xl` při hover)
- Scale efekt při hover (`scale-[1.02]`)
- Backdrop blur pro modernější vzhled

#### **ChatInput** (`components/ChatInput.tsx`)
- Gradient pozadí input oblasti
- Focus efekty s shadow a scale
- Backdrop blur

### 4. **Animace**

Všechny animace používají:
- **Duration**: 300-500ms (optimální pro UX)
- **Easing**: cubic-bezier pro plynulost
- **Delays**: Staggered animace u seznamů (50-100ms delay)

### 5. **Dark Mode**
- Kompletní podpora dark módu pro všechny gradienty
- Invertované gradienty v dark módu
- Smooth přechod mezi módy (500ms transition)

## Technické detaily

### CSS Custom Properties
```css
--background: 0 0% 100%;  /* Čistá bílá */
--foreground: 0 0% 5%;    /* Téměř černá */
--primary: 0 0% 10%;      /* Černá pro primary */
```

### Gradient Patterns
```css
/* Horizontal gradient */
bg-gradient-to-r from-white via-gray-50 to-white

/* Vertical gradient */
bg-gradient-to-b from-black via-gray-950 to-black

/* Diagonal gradient */
bg-gradient-to-br from-gray-50 via-white to-gray-50
```

### Animation Examples
```tsx
// Fade in
<div className="animate-fade-in">

// Staggered animation
<div
  className="animate-fade-in"
  style={{ animationDelay: `${index * 0.1}s` }}
>

// Hover effects
<div className="transition-all duration-300 hover:scale-105 hover:shadow-lg">
```

## Testování

Pro spuštění vývojového serveru:
```bash
cd frontend
npm install
npm run dev
```

Aplikace by měla běžet na `http://localhost:5173` s novým moderním designem.

## Kompatibilita

- ✅ React 18.3.1
- ✅ Tailwind CSS 3.4.14
- ✅ TypeScript 5.6.3
- ✅ Všechny moderní prohlížeče (Chrome, Firefox, Safari, Edge)

## Budoucí vylepšení

- Zvážit přidání micro-interactions (např. ripple efekt při kliknutí)
- Implementovat skeleton loadery s gradient shimmer
- Přidat page transitions
- Zvážit reduced motion pro accessibility
