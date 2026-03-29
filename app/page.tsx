export default function Home() {
  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-8">
      <div className="max-w-3xl text-center space-y-6">
        <h1 className="text-4xl font-bold tracking-tight text-foreground">
          Security Event Clustering
        </h1>
        <p className="text-lg text-muted-foreground">
          Deep Learning-based clustering for security events from firewall, IPS, DDoS, WAF, 
          web filter, mail protection, VPN and other subsystems.
        </p>
        <div className="bg-card border border-border rounded-lg p-6 text-left">
          <h2 className="text-xl font-semibold mb-4">Getting Started</h2>
          <p className="text-muted-foreground mb-4">
            This is a Python-based deep learning project. To run the clustering pipeline:
          </p>
          <pre className="bg-muted p-4 rounded-md text-sm overflow-x-auto">
{`cd scripts/security_clustering
uv sync
uv run python run_deep_clustering_demo.py`}
          </pre>
        </div>
        <div className="grid gap-4 md:grid-cols-2 text-left">
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Deep Clustering Methods</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>- DEC (Deep Embedded Clustering)</li>
              <li>- IDEC (Improved DEC)</li>
              <li>- VaDE (Variational Deep Embedding)</li>
              <li>- DCN (Deep Clustering Network)</li>
            </ul>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Supported Subsystems</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>- Firewall, IPS, DDoS Protection</li>
              <li>- WAF, Web Filter</li>
              <li>- Mail Protection, VPN</li>
              <li>- And more...</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  )
}
