import { CfnOutput, NestedStack,RemovalPolicy }  from 'aws-cdk-lib';
import {EngineVersion,Domain} from 'aws-cdk-lib/aws-opensearchservice';
import * as ec2 from 'aws-cdk-lib/aws-ec2';


export class OpenSearchStack extends NestedStack {
    domainEndpoint;
    domain;

/**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
constructor(scope, id, props) {
    super(scope, id, props);


    const devDomain = new Domain(this, 'Domain', {
        version: EngineVersion.OPENSEARCH_2_5,
        removalPolicy: RemovalPolicy.DESTROY,
        vpc:props.vpc,
        zoneAwareness: {
          enabled:true
        },
        // vpcSubnets:{
        //   subnets: props.subnets,
        // },
        capacity: {
            dataNodes: 2,
            dataNodeInstanceType:'r6g.large.search'
          },
        ebs: {
        volumeSize: 300,
        volumeType: ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD_GP3,
        },
      });
      this.domainEndpoint = devDomain.domainEndpoint;
      this.domain = devDomain;
      
}
}