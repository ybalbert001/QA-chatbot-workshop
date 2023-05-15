
import { NestedStack, CfnOutput,RemovalPolicy }  from 'aws-cdk-lib';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';


export class ALBStack extends NestedStack {

    dnsName = '';
    /**
     *
     * @param {Construct} scope
     * @param {string} id
     * @param {StackProps=} props
     */
    constructor(scope, id, props) {
      super(scope, id, props);
    
    const vpc = props.vpc;
    const privateSubnets = props.subnets;
    const domainEndpoint = props.opensearch_endpoint
    
    const alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
        vpc,
        internetFacing: true,
    });

     // Export the ALB DNS name so that it can be used by other stacks.
     this.dnsName = alb.loadBalancerDnsName;

    const listener = alb.addListener('HTTPListener',{
        port: 80,
        protocol: elbv2.Protocol.HTTP,
      });



    const opensearchTargetGroup = new elbv2.ApplicationTargetGroup(this, 'OpenSearchTargetGroup', {
        vpc,
        port: 443, 
        targetType:elbv2.TargetType.INSTANCE
      });
      
    listener.addTargetGroups('OpenSearchTargets', {
        targetGroups: [opensearchTargetGroup],
        port: 443,
      });

      listener.connections.allowDefaultPortFromAnyIpv4('Open to the world');


  
     




    }
}