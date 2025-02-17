// butterworthFilter.ts
export default class ButterworthFilter {
    private a0: number;
    private a1: number;
    private a2: number;
    private b1: number;
    private b2: number;
    private x1: number;
    private x2: number;
    private y1: number;
    private y2: number;
  
    /**
     * @param cutoff Fréquence de coupure en Hz (0.3 Hz dans votre cas)
     * @param sampleRate Fréquence d'échantillonnage en Hz (50 Hz dans votre cas)
     */
    constructor(cutoff: number, sampleRate: number) {
      const PI = Math.PI;
      const sqrt2 = Math.SQRT2;
  
      // Calcul préliminaire pour la transformation bilinéaire
      // On utilise la méthode pour un filtre Butterworth du second ordre.
      const wc = Math.tan((PI * cutoff) / sampleRate);
      const k1 = sqrt2 * wc;
      const k2 = wc * wc;
      const norm = 1 + k1 + k2;
  
      this.a0 = k2 / norm;
      this.a1 = 2 * this.a0;
      this.a2 = this.a0;
      this.b1 = 2 * (k2 - 1) / norm;
      this.b2 = (1 - k1 + k2) / norm;
  
      // Initialisation des états
      this.x1 = this.x2 = 0;
      this.y1 = this.y2 = 0;
    }
  
    /**
     * Applique le filtre à la nouvelle valeur d'entrée.
     * @param x Nouvelle mesure (exemple : donnée d'accélération sur un axe)
     * @returns La valeur filtrée
     */
    public process(x: number): number {
      const y = this.a0 * x + this.a1 * this.x1 + this.a2 * this.x2 - this.b1 * this.y1 - this.b2 * this.y2;
  
      // Mise à jour des anciens états
      this.x2 = this.x1;
      this.x1 = x;
      this.y2 = this.y1;
      this.y1 = y;
  
      return y;
    }
  }
  