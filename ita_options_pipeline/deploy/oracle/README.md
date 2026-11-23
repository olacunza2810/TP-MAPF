# Paper trading en Oracle Cloud (Always Free)

La VM corre el paper trading sola, de lunes a viernes, sin que tu compu esté
prendida. Lo único que hacés a mano es crear la cuenta, crear la VM, correr un
script y pegar las claves de Alpaca **paper**.

## Qué es gratis y qué no

- **Gratis para siempre (Always Free):** hasta 2 OCPU y 12 GB de RAM Ampere A1
  (Arm), o dos VM AMD de 1 GB; 200 GB de disco. Es lo único que vamos a usar.
- **Tarjeta:** te la piden para verificar identidad. No cobran salvo que pases
  la cuenta a *Pay As You Go*. Puede aparecer una autorización temporal. No
  aceptan prepagas ni débito con PIN.
- **Los primeros 30 días** traen USD 300 de crédito de prueba. Lo que no sea
  Always Free se paga con ese crédito y, al terminar la prueba, se borra; no se
  cobra.
- **Para que siga en cero:** no hagas *Upgrade* a Pay As You Go y creá todo con
  la etiqueta **Always Free-eligible**.
- **Riesgo real (no de plata):** Oracle puede **apagar** una VM Always Free que
  considere ociosa 7 días seguidos (CPU p95, red y, en A1, memoria por debajo
  de 20%). Si pasa, te avisan por mail y la volvés a prender desde la consola.

## 1. Cuenta (5–10 min)

1. Entrá a <https://www.oracle.com/cloud/free/> → *Start for free*.
2. Elegí la **home region** con cuidado: las VM Always Free sólo se crean ahí.
   Sugerencia: *Brazil East (Sao Paulo)* o *US East (Ashburn)*.
3. Completá mail, teléfono y tarjeta. **No** hagas upgrade.

## 2. VM (5 min)

En la consola: *Compute → Instances → Create instance*.

- **Image:** Canonical Ubuntu 24.04.
- **Shape:** *Ampere → VM.Standard.A1.Flex*, **1 OCPU y 6 GB** (dentro del
  límite de 2 / 12). Si dice *Out of host capacity*, probá otro *availability
  domain*, reintentá más tarde, o usá *VM.Standard.E2.1.Micro* (también Always
  Free; el script le agrega swap).
- **Networking:** dejá la VCN por defecto con **IP pública**.
- **SSH keys:** *Generate a key pair for me* → **Save private key** (guardá el
  archivo `.key` en tu compu; sin él no entrás).
- *Create*. Anotá la **Public IP**.

## 3. Conectarse y preparar (10 min)

En PowerShell de tu compu:

```powershell
ssh -i C:\ruta\a\tu-llave.key ubuntu@<IP_PUBLICA>
```

(Si Windows se queja de permisos de la llave: clic derecho → Propiedades →
Seguridad → dejá sólo tu usuario.)

Ya en la VM:

```bash
curl -fsSL https://raw.githubusercontent.com/olacunza2810/TP-MAPF/main/ita_options_pipeline/deploy/oracle/setup_vm.sh -o setup_vm.sh
bash setup_vm.sh
```

Instala todo, corre los tests y deja programada la corrida de lunes a viernes
a las 9:20 de Nueva York.

## 4. Claves de Alpaca paper (2 min)

```bash
nano ~/TP-MAPF/ita_options_pipeline/.env
```

Pegá (las mismas del `.env` de tu compu, **cuenta paper**):

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
```

Guardá (Ctrl+O, Enter, Ctrl+X) y:

```bash
chmod 600 ~/TP-MAPF/ita_options_pipeline/.env
cd ~/TP-MAPF/ita_options_pipeline
.venv/bin/python -m ita_options.pipeline paper-run --strategy rv_compite --max-cycles 1
```

Sin `--live` es una prueba en seco: verifica conexión y no manda órdenes.

## Día a día

- **No hay que hacer nada.** Cada rueda queda en `reportes/paper/` y el log en
  `reportes/paper/logs/AAAA-MM-DD.log`.
- **Ver cómo va:** `tail -f ~/TP-MAPF/ita_options_pipeline/reportes/paper/logs/$(date +%F).log`
- **Cortar todo:** `touch ~/TP-MAPF/ita_options_pipeline/KILL` (cierra posiciones
  en el ciclo siguiente). Para reanudar: `rm` de ese archivo.
- **Pausar la programación:** `crontab -r`.
- **Traer los resultados a tu compu** (desde la carpeta del pipeline):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\oracle\traer_resultados.ps1 -Ip <IP_PUBLICA> -Llave C:\ruta\a\tu-llave.key
```

## Al terminar el TP

*Compute → Instances → Terminate* (marcá borrar el boot volume). La cuenta
queda sin recursos y sin costo.
