# Brevo account email draft

Status: incomplete, disabled, and not deployed. The website still uses its existing Firebase account flow.

Resend rejected `kani.qd.je` as a free public domain in its Add domain screen. Do not enable email sending or add more DNS records to try to overcome that provider policy. The API key alone cannot enable sending from this domain.

The draft gateway module in `gateway/src/email.js` now prepares Firebase-generated verification and password-reset links for delivery through Brevo. It still needs dedicated route tests, frontend account/action-link handling, and an accepted, verified sender before activation. Registering a domain alone does not configure an inbox or activate sending.

Required secrets, kept outside source control:

- `BREVO_API_KEY`: Brevo API credential, not the SMTP key.
- `FIREBASE_SERVICE_ACCOUNT`: JSON credential for the `kelus-ide` project with the `firebaseauth.users.sendEmail` permission. This generates Firebase account-action links; the client API key is not sufficient.

`EMAIL_ENABLED` remains `false`. No live email has been sent. Do not enable it until the sender configuration, API integration tests, and browser action-link flow have passed. Set `EMAIL_FROM_ADDRESS` to the sender approved by Brevo and `EMAIL_FROM_NAME` to its display name before activation. Never paste API keys, private keys, or reset links into chat, logs, or source files.
